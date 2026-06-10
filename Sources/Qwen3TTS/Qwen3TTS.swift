import Foundation
import MLX
import MLXNN
import MLXFast
import AudioCommon

/// Errors thrown by streaming TTS synthesis.
public enum TTSError: Error, LocalizedError {
    case tokenizerNotLoaded
    case unknownLanguage(String)

    public var errorDescription: String? {
        switch self {
        case .tokenizerNotLoaded:
            return "Tokenizer not loaded. Call setTokenizer() first."
        case .unknownLanguage(let lang):
            return "Unknown language '\(lang)'"
        }
    }
}

/// Main Qwen3-TTS model for text-to-speech synthesis.
///
/// - Warning: This class is not thread-safe. Create separate instances for concurrent use.
public class Qwen3TTSModel {
    /// Default instruct text applied automatically for CustomVoice models when no explicit
    /// `--instruct` is provided. Prevents rambling output for short texts.
    public static let defaultInstruct = "Speak naturally."

    public let config: Qwen3TTSConfig
    public let talker: TalkerModel
    public let codePredictor: CodePredictorModel
    public let codecDecoder: SpeechTokenizerDecoder

    /// Speech tokenizer encoder for voice cloning (Base model only).
    /// Shares matched codebooks with `codecDecoder` when loaded from the same weights.
    public var codecEncoder: SpeechTokenizerEncoder?

    /// ECAPA-TDNN speaker encoder for voice cloning (Base model only)
    public let speakerEncoder: SpeakerEncoder

    /// Speaker configuration parsed from config.json (nil for Base model, populated for CustomVoice)
    public private(set) var speakerConfig: SpeakerConfig?

    /// Available speaker names (empty for Base model)
    public var availableSpeakers: [String] { speakerConfig?.availableSpeakers ?? [] }

    /// Model type identifier: "base", "custom_voice", or "voice_design"
    public var modelType: String { config.ttsModelType }

    /// Whether this is a VoiceDesign model that generates voices from text descriptions
    public var isVoiceDesign: Bool { config.ttsModelType == "voice_design" }

    /// Whether the codec encoder is loaded (Base models with encoder weights).
    public var hasCodecEncoder: Bool { codecEncoder != nil }

    /// Whether the model weights are loaded and ready for inference.
    var _isLoaded = true

    /// Computed suppress range for token sampling: (vocabSize - 1024, vocabSize).
    /// Suppresses the last 1024 tokens of the codec vocabulary (special tokens) except EOS.
    /// Matches reference implementation's dynamic calculation instead of hardcoded (2048, 3072).
    private var codecSuppressRange: (Int, Int) {
        let vocabSize = config.talker.codecVocabSize
        return (vocabSize - 1024, vocabSize)
    }

    private var tokenizer: Qwen3Tokenizer?

    /// Compiled talker generation step (28-layer transformer + codec head) for kernel fusion.
    /// Fuses ~420 Metal kernel dispatches per step into fewer optimized kernels.
    ///
    /// Uses shapeless=true: handles growing KV cache without recompilation.
    /// RoPE offset is passed as a regular function input (compile treats inputs as variables).
    /// Batch dimension uses -1 reshapes so the same compiled graph works for any batch size.
    private var compiledTalkerStep: (([MLXArray]) -> [MLXArray])?

    /// Compiled code predictor transformer (layers + norm, no lm_head) for kernel fusion.
    /// Used for groups 1-14 of per-timestep code prediction (seqLen=1 with cache).
    ///
    /// Uses shapeless=false: one compiled graph per cache size (14 sizes, compiled once during warmup).
    /// Each group i always has cache seqLen=i+2, so compiled graphs are reused across timesteps.
    ///
    /// Talker is compiled with shapeless=true — RoPE offset passed as regular MLXArray input,
    /// growing KV cache handled by shapeless mode, batch dim uses -1 reshapes.
    private var compiledCPTransformer: (([MLXArray]) -> [MLXArray])?

    public init(config: Qwen3TTSConfig = .base06B) {
        self.config = config
        self.talker = TalkerModel(config: config.talker)
        self.codePredictor = CodePredictorModel(config: config.codePredictor)
        self.codecDecoder = SpeechTokenizerDecoder(config: config.speechTokenizerDecoder)
        self.speakerEncoder = SpeakerEncoder(encDim: config.speakerEncoderDim)
    }

    public func setTokenizer(_ tokenizer: Qwen3Tokenizer) {
        self.tokenizer = tokenizer
    }

    /// Synthesize speech from text
    /// - Parameters:
    ///   - text: Input text to synthesize
    ///   - language: Language tag (e.g., "english", "chinese")
    ///   - speaker: Speaker voice name (requires CustomVoice model, e.g., "vivian", "ryan")
    ///   - instruct: Instruction text for style control (requires CustomVoice model, e.g., "Speak cheerfully")
    ///   - sampling: Sampling configuration
    /// - Returns: Audio samples at 24kHz
    public func synthesize(
        text: String,
        language: String = "english",
        speaker: String? = nil,
        instruct: String? = nil,
        sampling: SamplingConfig = .default
    ) -> [Float] {
        guard let tokenizer = tokenizer else {
            fatalError("Tokenizer not loaded. Call setTokenizer() first.")
        }

        // Resolve speaker → token ID and optional language override
        let (speakerTokenId, effectiveLanguage) = resolveSpeaker(speaker, language: language)

        // Auto-apply default instruct for CustomVoice when none provided.
        // VoiceDesign: instruct IS the voice description — don't override with default.
        let effectiveInstruct: String?
        if config.ttsModelType == "voice_design" {
            effectiveInstruct = instruct  // voice description passed directly, nil means no instruct
        } else {
            effectiveInstruct = instruct ?? (speakerConfig != nil ? Self.defaultInstruct : nil)
        }

        guard let langId = resolveLanguageId(for: effectiveLanguage) else {
            print("Warning: Unknown language '\(effectiveLanguage)', defaulting to English")
            return synthesize(text: text, language: "english", speaker: speaker, sampling: sampling)
        }

        let t0 = CFAbsoluteTimeGetCurrent()

        // Stage 1: Prepare text tokens and codec prefix
        let textTokens = prepareTextTokens(text: text, tokenizer: tokenizer)
        let codecPrefixTokens = buildCodecPrefix(languageId: langId, speakerTokenId: speakerTokenId)
        let instructTokens = effectiveInstruct.map { prepareInstructTokens(instruct: $0, tokenizer: tokenizer) }

        // Stage 2: Build input embeddings with element-wise text+codec overlay
        // VoiceDesign uses non-streaming mode (all text in prefill) matching the Python default.
        // Voice cloning and streaming synthesis use streaming mode (one text token per codec step).
        let useNonStreaming = (config.ttsModelType == "voice_design" && instruct != nil)
        let (prefillEmbeds, trailingTextHidden, ttsPadEmbed) = buildPrefillEmbeddings(
            textTokens: textTokens, codecPrefixTokens: codecPrefixTokens, instructTokens: instructTokens,
            nonStreamingMode: useNonStreaming)

        eval(prefillEmbeds, trailingTextHidden, ttsPadEmbed)
        let t1 = CFAbsoluteTimeGetCurrent()

        // Stage 3: Autoregressive generation with per-step code predictor
        // Apply dynamic max token cap based on text length to prevent runaways.
        // Python reference uses max(75, tokenCount * 6). We use a more generous multiplier
        // (10x vs 6x) and much higher floor (1000 vs 75) to accommodate anime dialogue
        // which often has dramatic pauses, emotional delivery, and longer sentences.
        // At 12.5 Hz codec rate, 1000 tokens ≈ 80s audio — enough for most dialogue blocks.
        var effectiveSampling = sampling
        let textContentTokens = max(textTokens.count - 8, 1)  // subtract ChatML overhead
        let dynamicMaxTokens = max(1000, textContentTokens * 10)
        effectiveSampling.maxTokens = min(effectiveSampling.maxTokens, dynamicMaxTokens)
        // Note: repetition penalty stays at default 1.05 for CustomVoice/standard synthesis.
        // Only ICL voice cloning (synthesizeWithVoiceClone) elevates to 1.5.
        // The Python reference uses 1.05 for non-ICL generation.

        let (allCodebooks, numFrames) = generateWithCodePredictor(
            prefillEmbeds: prefillEmbeds,
            trailingTextHidden: trailingTextHidden,
            ttsPadEmbed: ttsPadEmbed,
            sampling: effectiveSampling)

        eval(allCodebooks)
        let t2 = CFAbsoluteTimeGetCurrent()

        guard numFrames > 0 else {
            print("Warning: Talker generated no tokens")
            return []
        }

        // Stage 4: Codec decode to waveform
        let outputSamples = numFrames * 1920
        print("  Decoding \(numFrames) frames -> \(outputSamples) samples (\(String(format: "%.1f", Double(outputSamples) / 24000.0))s)...")
        let waveform = codecDecoder.decode(codes: allCodebooks)
        let t3 = CFAbsoluteTimeGetCurrent()

        // Stage 5: Trim tonal tail artifacts
        let trimmedWaveform = trimTonalTail(waveform)

        let audioDur = Double(trimmedWaveform.count) / 24000.0
        print("  Timing: embed=\(String(format: "%.3f", t1-t0))s | " +
              "generate=\(String(format: "%.3f", t2-t1))s (\(numFrames) steps, " +
              "\(String(format: "%.0f", (t2-t1)/Double(numFrames)*1000))ms/step) | " +
              "decode=\(String(format: "%.3f", t3-t2))s | " +
              "total=\(String(format: "%.3f", t3-t0))s | " +
              "audio=\(String(format: "%.2f", audioDur))s | " +
              "RTF=\(String(format: "%.2f", (t3-t0)/audioDur))")

        return trimmedWaveform
    }

    // MARK: - VoiceDesign Synthesis

    /// Synthesize speech using a natural-language voice description (VoiceDesign model only).
    ///
    /// Generates custom character voices from text descriptions across seven dimensions:
    /// gender, age, pitch, pace, emotion, vocal characteristics, and use case.
    /// Recommended description length: 15–40 words. Descriptions must be in English or Chinese.
    ///
    /// - Parameters:
    ///   - text: Input text to synthesize
    ///   - voiceDescription: Natural-language voice description (e.g., "A cheerful young female voice with bright energy")
    ///   - language: Language tag for the output speech (e.g., "english", "chinese")
    ///   - sampling: Sampling configuration
    /// - Returns: Audio samples at 24kHz
    public func synthesizeVoiceDesign(
        text: String,
        voiceDescription: String,
        language: String = "english",
        sampling: SamplingConfig = .voiceDesign
    ) -> [Float] {
        synthesize(text: text, language: language, speaker: nil, instruct: voiceDescription, sampling: sampling)
    }

    /// Stream synthesis using a natural-language voice description (VoiceDesign model only).
    ///
    /// - Parameters:
    ///   - text: Input text to synthesize
    ///   - voiceDescription: Natural-language voice description
    ///   - language: Language tag for the output speech
    ///   - sampling: Sampling configuration
    ///   - streaming: Streaming configuration (chunk sizes, decoder context)
    /// - Returns: An async stream of `AudioChunk` values
    public func synthesizeVoiceDesignStream(
        text: String,
        voiceDescription: String,
        language: String = "english",
        sampling: SamplingConfig = .voiceDesign,
        streaming: StreamingConfig = .default
    ) -> AsyncThrowingStream<AudioChunk, Error> {
        synthesizeStream(text: text, language: language, speaker: nil, instruct: voiceDescription,
                         sampling: sampling, streaming: streaming)
    }

    // MARK: - Voice Cloning (x-vector mode)

    /// Synthesize speech cloning a reference speaker's voice.
    ///
    /// Extracts a speaker embedding from reference audio using the ECAPA-TDNN speaker encoder,
    /// then injects it into the codec prefix embedding between the think tokens and pad/bos.
    ///
    /// - Parameters:
    ///   - text: Input text to synthesize
    ///   - referenceAudio: Reference speaker audio samples (any sample rate, will be resampled to 24kHz)
    ///   - referenceSampleRate: Sample rate of reference audio
    ///   - language: Language tag (e.g., "english", "chinese")
    ///   - sampling: Sampling configuration
    /// - Returns: Audio samples at 24kHz
    public func synthesizeWithVoiceClone(
        text: String,
        referenceAudio: [Float],
        referenceSampleRate: Int = 24000,
        language: String = "english",
        sampling: SamplingConfig = .default
    ) -> [Float] {
        guard let tokenizer = tokenizer else {
            fatalError("Tokenizer not loaded. Call setTokenizer() first.")
        }

        guard let langId = resolveLanguageId(for: language) else {
            print("Warning: Unknown language '\(language)', defaulting to English")
            return synthesizeWithVoiceClone(
                text: text, referenceAudio: referenceAudio,
                referenceSampleRate: referenceSampleRate, language: "english", sampling: sampling)
        }

        let t0 = CFAbsoluteTimeGetCurrent()

        // Extract speaker embedding from reference audio
        let mels = SpeakerMel.compute(audio: referenceAudio, sampleRate: referenceSampleRate)
        let speakerEmbed = speakerEncoder(mels)  // [1, 1024]
        eval(speakerEmbed)
        print("  Speaker embedding extracted: \(speakerEmbed.shape)")

        // Stage 1: Prepare text tokens and codec prefix (no speaker token ID — using embedding)
        let textTokens = prepareTextTokens(text: text, tokenizer: tokenizer)
        let codecPrefixTokens = buildCodecPrefix(languageId: langId)

        // Stage 2: Build input embeddings with speaker embedding injection
        let (prefillEmbeds, trailingTextHidden, ttsPadEmbed) = buildPrefillEmbeddings(
            textTokens: textTokens, codecPrefixTokens: codecPrefixTokens,
            speakerEmbedding: speakerEmbed)

        eval(prefillEmbeds, trailingTextHidden, ttsPadEmbed)
        let t1 = CFAbsoluteTimeGetCurrent()

        // Stage 3: Autoregressive generation with per-step code predictor
        // Voice cloning benefits from elevated repetition penalty (1.5) to suppress
        // codec token repetition that causes audible stuttering. The default 1.05 is
        // too gentle for cloned-voice generation. Match the ICL pipeline's setting.
        var cloneSampling = sampling
        cloneSampling.repetitionPenalty = max(sampling.repetitionPenalty, 1.5)

        let (allCodebooks, numFrames) = generateWithCodePredictor(
            prefillEmbeds: prefillEmbeds,
            trailingTextHidden: trailingTextHidden,
            ttsPadEmbed: ttsPadEmbed,
            sampling: cloneSampling)

        eval(allCodebooks)
        let t2 = CFAbsoluteTimeGetCurrent()

        guard numFrames > 0 else {
            print("Warning: Talker generated no tokens")
            return []
        }

        // Stage 4: Codec decode to waveform
        let outputSamples = numFrames * 1920
        print("  Decoding \(numFrames) frames -> \(outputSamples) samples (\(String(format: "%.1f", Double(outputSamples) / 24000.0))s)...")
        let waveform = codecDecoder.decode(codes: allCodebooks)
        let t3 = CFAbsoluteTimeGetCurrent()

        // Stage 5: Trim tonal tail artifacts
        let trimmedWaveform = trimTonalTail(waveform)

        let audioDur = Double(trimmedWaveform.count) / 24000.0
        print("  Voice clone timing: embed=\(String(format: "%.3f", t1-t0))s | " +
              "generate=\(String(format: "%.3f", t2-t1))s (\(numFrames) steps, " +
              "\(String(format: "%.0f", (t2-t1)/Double(numFrames)*1000))ms/step) | " +
              "decode=\(String(format: "%.3f", t3-t2))s | " +
              "total=\(String(format: "%.3f", t3-t0))s | " +
              "audio=\(String(format: "%.2f", audioDur))s | " +
              "RTF=\(String(format: "%.2f", (t3-t0)/audioDur))")

        return trimmedWaveform
    }

    // MARK: - Streaming Synthesis

    /// Synthesize speech as a stream of audio chunks with low first-packet latency.
    ///
    /// The architecture is fully causal (Talker, Code Predictor, Mimi decoder all use causal
    /// attention/convolutions), so streaming produces the same quality as batch synthesis.
    ///
    /// - Parameters:
    ///   - text: Input text to synthesize
    ///   - language: Language tag (e.g., "english", "chinese")
    ///   - speaker: Speaker voice name (requires CustomVoice model)
    ///   - instruct: Instruction text for style control (requires CustomVoice model)
    ///   - sampling: Sampling configuration
    ///   - streaming: Streaming configuration (chunk sizes, decoder context)
    /// - Returns: An async stream of `AudioChunk` values
    public func synthesizeStream(
        text: String,
        language: String = "english",
        speaker: String? = nil,
        instruct: String? = nil,
        sampling: SamplingConfig = .default,
        streaming: StreamingConfig = .default
    ) -> AsyncThrowingStream<AudioChunk, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    try self.runStreamingGeneration(
                        text: text,
                        language: language,
                        speaker: speaker,
                        instruct: instruct,
                        sampling: sampling,
                        streaming: streaming,
                        continuation: continuation)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    /// Internal streaming generation loop. Same structure as `synthesize()` but emits audio
    /// chunks via the continuation as soon as enough frames are accumulated.
    private func runStreamingGeneration(
        text: String,
        language: String,
        speaker: String?,
        instruct: String?,
        sampling: SamplingConfig,
        streaming: StreamingConfig,
        continuation: AsyncThrowingStream<AudioChunk, Error>.Continuation
    ) throws {
        guard let tokenizer = tokenizer else {
            throw TTSError.tokenizerNotLoaded
        }

        let (speakerTokenId, effectiveLanguage) = resolveSpeaker(speaker, language: language)

        // Auto-apply default instruct: VoiceDesign passes instruct as-is (voice description)
        let effectiveInstruct: String?
        if config.ttsModelType == "voice_design" {
            effectiveInstruct = instruct
        } else {
            effectiveInstruct = instruct ?? (speakerConfig != nil ? Self.defaultInstruct : nil)
        }

        guard let langId = resolveLanguageId(for: effectiveLanguage) else {
            throw TTSError.unknownLanguage(effectiveLanguage)
        }

        let t0 = CFAbsoluteTimeGetCurrent()

        // Seed RNG once at the start of generation (not per-token).
        // Per-token seeding caused runaway generation for certain seed values.
        if let seed = sampling.seed {
            seedSamplingRNG(seed)
        }

        let safeMaxTokens = min(sampling.maxTokens, 1500)
        let samplesPerFrame = 1920  // 24000 / 12.5

        // Stage 1: Prepare embeddings (identical to synthesize)
        let textTokens = prepareTextTokens(text: text, tokenizer: tokenizer)
        let codecPrefixTokens = buildCodecPrefix(languageId: langId, speakerTokenId: speakerTokenId)
        let instructTokens = effectiveInstruct.map { prepareInstructTokens(instruct: $0, tokenizer: tokenizer) }
        let useNonStreaming = (config.ttsModelType == "voice_design" && instruct != nil)
        let (prefillEmbeds, trailingTextHidden, ttsPadEmbed) = buildPrefillEmbeddings(
            textTokens: textTokens, codecPrefixTokens: codecPrefixTokens, instructTokens: instructTokens,
            nonStreamingMode: useNonStreaming)
        eval(prefillEmbeds, trailingTextHidden, ttsPadEmbed)

        // Stage 2: Autoregressive generation with chunked decode + emit
        let cpSamplingConfig = SamplingConfig(temperature: sampling.temperature, topK: sampling.topK)
        let prefillLen = prefillEmbeds.dim(1)

        // Prefill
        var (logits, hiddenStates, newCache) = talker(
            inputsEmbeds: prefillEmbeds,
            offset: MLXArray(Int32(0)),
            cache: nil)
        var talkerCache = newCache

        // Sample first token
        let lastLogits = logits[0..., (prefillLen - 1)..<prefillLen, 0...]
        var nextToken = sampleToken(
            logits: lastLogits,
            config: sampling,
            generatedTokens: [],
            suppressRange: codecSuppressRange,
            eosTokenId: CodecTokens.codecEos)

        if nextToken == Int32(CodecTokens.codecEos) {
            let chunk = AudioChunk(
                samples: [], sampleRate: 24000, frameIndex: 0,
                isFinal: true,
                elapsedTime: CFAbsoluteTimeGetCurrent() - t0)
            continuation.yield(chunk)
            return
        }

        var generatedFirstCodebook: [Int32] = [nextToken]
        var generatedAllCodebooks: [[Int32]] = (0..<config.codePredictor.numCodeGroups).map { _ in [] }
        generatedAllCodebooks[0].append(nextToken)

        // Code predictor for first timestep
        let lastHidden = hiddenStates[0..., (prefillLen - 1)..<prefillLen, 0...]
        var codeTokens = predictCodebooksForTimestep(
            hiddenState: lastHidden,
            firstCodebookToken: nextToken,
            cpSamplingConfig: cpSamplingConfig)
        for (i, token) in codeTokens.enumerated() {
            generatedAllCodebooks[i + 1].append(token)
        }

        var trailingIdx = 0
        var step = prefillLen
        var emittedFrames = 0
        var emittedFinal = false

        var nextEmitThreshold = streaming.firstChunkFrames

        // Emit immediately if prefill already produced enough frames (e.g., firstChunkFrames=1)
        if generatedFirstCodebook.count >= nextEmitThreshold {
            let chunk = decodeAndEmitChunk(
                allCodebooks: generatedAllCodebooks,
                chunkStart: 0,
                chunkEnd: generatedFirstCodebook.count,
                decoderLeftContext: streaming.decoderLeftContext,
                samplesPerFrame: samplesPerFrame)
            let audioChunk = AudioChunk(
                samples: chunk,
                sampleRate: 24000,
                frameIndex: 0,
                isFinal: false,
                elapsedTime: CFAbsoluteTimeGetCurrent() - t0)
            continuation.yield(audioChunk)
            emittedFrames = generatedFirstCodebook.count
            nextEmitThreshold = emittedFrames + streaming.chunkFrames
        }

        // Autoregressive generation loop
        for iterIdx in 1..<safeMaxTokens {
            // Text side
            let textEmbed: MLXArray
            let trailingLen = trailingTextHidden.dim(1)
            if trailingIdx < trailingLen {
                textEmbed = trailingTextHidden[0..., trailingIdx..<(trailingIdx + 1), 0...]
                trailingIdx += 1
            } else {
                textEmbed = ttsPadEmbed
            }

            // Codec side
            let codecEmbed = talker.embedCodec(
                MLXArray([nextToken]).expandedDimensions(axis: 0))
                + codePredictor.batchEmbedAllGroups(codeTokens)

            let stepEmbeds = textEmbed + codecEmbed

            (logits, hiddenStates, newCache) = executeTalkerStep(
                embeds: stepEmbeds, offset: step, cache: talkerCache)
            talkerCache = newCache

            nextToken = sampleToken(
                logits: logits,
                config: sampling,
                generatedTokens: generatedFirstCodebook,
                suppressRange: codecSuppressRange,
                eosTokenId: CodecTokens.codecEos)

            let isEos = nextToken == Int32(CodecTokens.codecEos)

            if !isEos {
                generatedFirstCodebook.append(nextToken)
                generatedAllCodebooks[0].append(nextToken)

                let stepHidden = hiddenStates
                codeTokens = predictCodebooksForTimestep(
                    hiddenState: stepHidden,
                    firstCodebookToken: nextToken,
                    cpSamplingConfig: cpSamplingConfig)
                for (i, token) in codeTokens.enumerated() {
                    generatedAllCodebooks[i + 1].append(token)
                }
            }

            step += 1
            let totalFrames = generatedFirstCodebook.count

            // Emit chunk when we have enough frames or on EOS/max
            let shouldEmit = isEos || totalFrames >= nextEmitThreshold || iterIdx == safeMaxTokens - 1
            if shouldEmit && totalFrames > emittedFrames {
                let chunkFrameStart = emittedFrames
                let chunkFrameEnd = totalFrames

                let chunk = decodeAndEmitChunk(
                    allCodebooks: generatedAllCodebooks,
                    chunkStart: chunkFrameStart,
                    chunkEnd: chunkFrameEnd,
                    decoderLeftContext: streaming.decoderLeftContext,
                    samplesPerFrame: samplesPerFrame)

                let isFinalChunk = isEos || iterIdx == safeMaxTokens - 1
                let audioChunk = AudioChunk(
                    samples: chunk,
                    sampleRate: 24000,
                    frameIndex: chunkFrameStart,
                    isFinal: isFinalChunk,
                    elapsedTime: CFAbsoluteTimeGetCurrent() - t0)
                continuation.yield(audioChunk)
                if isFinalChunk { emittedFinal = true }

                emittedFrames = chunkFrameEnd
                // After first emit, use regular chunk size
                nextEmitThreshold = emittedFrames + streaming.chunkFrames
            }

            if isEos { break }

            if iterIdx % 50 == 0 {
                let estSec = Double(generatedFirstCodebook.count) / 12.5
                print("  Streaming: \(generatedFirstCodebook.count) tokens (~\(String(format: "%.1f", estSec))s audio)...")
            }
        }

        let numFrames = generatedFirstCodebook.count
        if numFrames >= safeMaxTokens && nextToken != Int32(CodecTokens.codecEos) {
            let estSec = Double(numFrames) / 12.5
            print("Warning: Hit safety limit of \(safeMaxTokens) tokens (~\(String(format: "%.1f", estSec))s audio).")
        }

        // Emit remaining frames if any
        if emittedFrames < numFrames {
            let chunk = decodeAndEmitChunk(
                allCodebooks: generatedAllCodebooks,
                chunkStart: emittedFrames,
                chunkEnd: numFrames,
                decoderLeftContext: streaming.decoderLeftContext,
                samplesPerFrame: samplesPerFrame)
            let audioChunk = AudioChunk(
                samples: chunk,
                sampleRate: 24000,
                frameIndex: emittedFrames,
                isFinal: true,
                elapsedTime: CFAbsoluteTimeGetCurrent() - t0)
            continuation.yield(audioChunk)
            emittedFinal = true
        }

        // If EOS arrived with no new frames, emit a final sentinel
        if !emittedFinal {
            let audioChunk = AudioChunk(
                samples: [],
                sampleRate: 24000,
                frameIndex: emittedFrames,
                isFinal: true,
                elapsedTime: CFAbsoluteTimeGetCurrent() - t0)
            continuation.yield(audioChunk)
        }
    }

    /// Decode a chunk of codec frames to audio samples, using left context for decoder quality.
    ///
    /// Builds `[1, 16, contextFrames + chunkFrames]` from accumulated codebooks, runs the codec
    /// decoder, trims left-context and zero-pad samples, and returns Float PCM.
    ///
    /// The codec decoder (ConvNeXt kernel=7 after 2x pre-upsample) requires >= 4 input frames.
    /// When fewer real frames are available (e.g., 1-frame first chunk with no context), zeros
    /// are prepended as left padding. The decoder is fully causal (left-padded convolutions),
    /// so zero-padding produces silence that doesn't affect the real frames' output.
    private func decodeAndEmitChunk(
        allCodebooks: [[Int32]],
        chunkStart: Int,
        chunkEnd: Int,
        decoderLeftContext: Int,
        samplesPerFrame: Int
    ) -> [Float] {
        let contextStart = max(chunkStart - decoderLeftContext, 0)
        let actualContext = chunkStart - contextStart

        // Build [1, 16, contextFrames + chunkFrames] from accumulated codebooks
        let numGroups = allCodebooks.count
        let frameRange = contextStart..<chunkEnd
        var codebookArrays: [MLXArray] = []
        for g in 0..<numGroups {
            let slice = Array(allCodebooks[g][frameRange])
            codebookArrays.append(MLXArray(slice).expandedDimensions(axis: 0))  // [1, T]
        }
        var codes = stacked(codebookArrays, axis: 1)  // [1, 16, T]

        // Zero-pad if fewer than 4 frames (codec decoder minimum for ConvNeXt kernel=7)
        let minDecodeFrames = 4
        let realFrames = codes.dim(2)
        let zeroPadFrames = max(minDecodeFrames - realFrames, 0)
        if zeroPadFrames > 0 {
            let pad = MLXArray.zeros([1, numGroups, zeroPadFrames]).asType(.int32)
            codes = concatenated([pad, codes], axis: 2)  // prepend zeros on left
        }

        // Decode through codec (uses compiled path when available)
        let waveform = codecDecoder.executeDecoder(codes)  // [1, T_samples, 1]

        // Keep the last `realChunkFrames * samplesPerFrame` samples from the decoder output.
        // The decoder has a ~2880-sample startup overhead (causal conv warmup), so trimming
        // from the left by `(zeroPad + context) * samplesPerFrame` can overshoot. Instead,
        // compute the desired output size and trim from the right side of the waveform.
        let realChunkFrames = chunkEnd - chunkStart
        let expectedKept = realChunkFrames * samplesPerFrame
        let totalSamples = waveform.dim(1)
        let trimSamples = max(0, totalSamples - expectedKept)
        let kept = waveform[0..., trimSamples..<totalSamples, 0...]

        let flat = kept.squeezed()
        eval(flat)
        return flat.asArray(Float.self)
    }

    // MARK: - Batch Synthesis

    /// Synthesize speech from multiple texts in parallel using batched generation.
    ///
    /// All items generate tokens in lockstep. Items that finish early (hit EOS) receive
    /// padding tokens. Generation stops when all items are done or the safety cap is reached.
    ///
    /// Texts are sorted by length before batching so similar-length items are grouped together,
    /// minimizing wasted compute from padding. Results are returned in the original input order.
    ///
    /// **Memory:** Each item uses ~110 MB KV cache per 1000 tokens. B=4 at 1500 tokens ≈ 660 MB.
    ///
    /// **Limitations:**
    /// - Repetition penalty is not applied in batch mode (requires per-item token history).
    /// - Items with very different output lengths waste compute on padding steps.
    ///   If one item fails to hit EOS, all items in the batch run to the safety cap.
    ///
    /// - Parameters:
    ///   - texts: Array of texts to synthesize
    ///   - language: Language tag (e.g., "english", "chinese")
    ///   - sampling: Sampling configuration
    ///   - maxBatchSize: Maximum items per batch (default 4)
    /// - Returns: Array of audio samples at 24kHz, one per input text (same order as input)
    public func synthesizeBatch(
        texts: [String],
        language: String = "english",
        instruct: String? = nil,
        sampling: SamplingConfig = .default,
        maxBatchSize: Int = 4
    ) -> [[Float]] {
        guard !texts.isEmpty else { return [] }

        // Single item: delegate to existing method for zero overhead
        if texts.count == 1 {
            return [synthesize(text: texts[0], language: language, instruct: instruct, sampling: sampling)]
        }

        guard let tokenizer = tokenizer else {
            fatalError("Tokenizer not loaded. Call setTokenizer() first.")
        }

        // Auto-apply default instruct: VoiceDesign passes instruct as-is (voice description)
        let effectiveInstruct: String?
        if config.ttsModelType == "voice_design" {
            effectiveInstruct = instruct
        } else {
            effectiveInstruct = instruct ?? (speakerConfig != nil ? Self.defaultInstruct : nil)
        }

        guard let langId = resolveLanguageId(for: language) else {
            print("Warning: Unknown language '\(language)', defaulting to English")
            return synthesizeBatch(texts: texts, language: "english", instruct: instruct, sampling: sampling, maxBatchSize: maxBatchSize)
        }

        // Sort texts by length to group similar-length items together.
        // This minimizes padding waste: if one batch has all short texts and another
        // has all long texts, no short text is forced to wait for a long one.
        let indexed = texts.enumerated().map { ($0.offset, $0.element) }
        let sorted = indexed.sorted { $0.1.count < $1.1.count }
        let sortedTexts = sorted.map { $0.1 }
        let originalIndices = sorted.map { $0.0 }
        let instructTokens = effectiveInstruct.map { prepareInstructTokens(instruct: $0, tokenizer: tokenizer) }

        // Process in chunks if exceeding maxBatchSize
        var sortedResults: [[Float]]
        if sortedTexts.count > maxBatchSize {
            sortedResults = []
            for chunkStart in stride(from: 0, to: sortedTexts.count, by: maxBatchSize) {
                let chunkEnd = min(chunkStart + maxBatchSize, sortedTexts.count)
                let chunk = Array(sortedTexts[chunkStart..<chunkEnd])
                let chunkResults = synthesizeBatchInternal(texts: chunk, langId: langId, instructTokens: instructTokens, tokenizer: tokenizer, sampling: sampling)
                sortedResults.append(contentsOf: chunkResults)
            }
        } else {
            sortedResults = synthesizeBatchInternal(texts: sortedTexts, langId: langId, instructTokens: instructTokens, tokenizer: tokenizer, sampling: sampling)
        }

        // Restore original order
        var results = [[Float]](repeating: [], count: texts.count)
        for (sortedIdx, origIdx) in originalIndices.enumerated() {
            results[origIdx] = sortedResults[sortedIdx]
        }
        return results
    }

    /// Internal batch synthesis for a single chunk (already sorted, within maxBatchSize).
    private func synthesizeBatchInternal(
        texts: [String],
        langId: Int,
        instructTokens: [Int]?,
        tokenizer: Qwen3Tokenizer,
        sampling: SamplingConfig
    ) -> [[Float]] {
        let t0 = CFAbsoluteTimeGetCurrent()
        let batchSize = texts.count

        // Stage 1: Prepare per-item data
        let codecPrefixTokens = buildCodecPrefix(languageId: langId)

        var prefills: [MLXArray] = []
        var trailings: [MLXArray] = []
        var padEmbeds: [MLXArray] = []

        let useNonStreaming = (config.ttsModelType == "voice_design" && instructTokens != nil)
        for text in texts {
            let textTokens = prepareTextTokens(text: text, tokenizer: tokenizer)
            let (prefill, trailing, padEmbed) = buildPrefillEmbeddings(
                textTokens: textTokens, codecPrefixTokens: codecPrefixTokens, instructTokens: instructTokens,
                nonStreamingMode: useNonStreaming)
            prefills.append(prefill)
            trailings.append(trailing)
            padEmbeds.append(padEmbed)
        }

        // All prefills have the same length — stack directly
        let batchPrefill = concatenated(prefills, axis: 0)  // [B, prefillLen, D]
        let ttsPadEmbed = padEmbeds[0]  // All pad embeds are the same — use first

        // Pre-pad trailing texts to max length
        let maxTrailingLen = trailings.map { $0.dim(1) }.max()!
        var paddedTrailings: [MLXArray] = []
        for trailing in trailings {
            let trailLen = trailing.dim(1)
            if trailLen < maxTrailingLen {
                let padCount = maxTrailingLen - trailLen
                let padding = broadcast(ttsPadEmbed, to: [1, padCount, config.talker.hiddenSize])
                paddedTrailings.append(concatenated([trailing, padding], axis: 1))
            } else {
                paddedTrailings.append(trailing)
            }
        }
        let batchTrailing = concatenated(paddedTrailings, axis: 0)  // [B, maxTrailingLen, D]

        eval(batchPrefill, batchTrailing)
        let t1 = CFAbsoluteTimeGetCurrent()

        // Stage 2: Batch generation
        let (allCodebooksList, frameCounts) = generateBatchWithCodePredictor(
            batchPrefill: batchPrefill,
            batchTrailing: batchTrailing,
            ttsPadEmbed: ttsPadEmbed,
            sampling: sampling)

        let t2 = CFAbsoluteTimeGetCurrent()

        // Log padding waste: how many steps each item wasted after hitting EOS
        let maxFrames = frameCounts.max() ?? 0
        if maxFrames > 0 {
            let wastedSteps = frameCounts.map { maxFrames - $0 }
            let totalWaste = wastedSteps.reduce(0, +)
            let wasteRatio = Double(totalWaste) / Double(maxFrames * batchSize)
            if wasteRatio > 0.3 {
                print("  Warning: \(Int(wasteRatio * 100))% padding waste " +
                      "(items finished at: \(frameCounts.map { String($0) }.joined(separator: ", ")) steps). " +
                      "Batch similar-length texts for better efficiency.")
            }
        }

        // Stage 3: Decode each item
        var results: [[Float]] = []
        for i in 0..<batchSize {
            let numFrames = frameCounts[i]
            if numFrames == 0 {
                print("  Item \(i): no tokens generated")
                results.append([])
                continue
            }
            let codes = allCodebooksList[i]  // [1, 16, Ti]
            let outputSamples = numFrames * 1920
            print("  Item \(i): decoding \(numFrames) frames -> \(outputSamples) samples (\(String(format: "%.1f", Double(outputSamples) / 24000.0))s)...")
            let waveform = codecDecoder.decode(codes: codes)
            results.append(trimTonalTail(waveform))
        }
        let t3 = CFAbsoluteTimeGetCurrent()

        let totalAudio = results.reduce(0.0) { $0 + Double($1.count) / 24000.0 }
        let totalFrames = frameCounts.reduce(0, +)
        print("  Batch timing: embed=\(String(format: "%.3f", t1-t0))s | " +
              "generate=\(String(format: "%.3f", t2-t1))s (\(totalFrames) total steps, " +
              "\(batchSize) items) | " +
              "decode=\(String(format: "%.3f", t3-t2))s | " +
              "total=\(String(format: "%.3f", t3-t0))s | " +
              "audio=\(String(format: "%.2f", totalAudio))s | " +
              "RTF=\(String(format: "%.2f", (t3-t0)/max(totalAudio, 0.001)))")

        return results
    }

    // MARK: - Batch Generation Loop

    /// Batch autoregressive generation: all B items in lockstep.
    private func generateBatchWithCodePredictor(
        batchPrefill: MLXArray,
        batchTrailing: MLXArray,
        ttsPadEmbed: MLXArray,
        sampling: SamplingConfig
    ) -> (allCodebooksList: [MLXArray], frameCounts: [Int]) {
        // Seed RNG once at the start of generation (not per-token).
        // Per-token seeding caused runaway generation for certain seed values.
        if let seed = sampling.seed {
            seedSamplingRNG(seed)
        }

        let batchSize = batchPrefill.dim(0)
        let safeMaxTokens = min(sampling.maxTokens, 1500)
        let maxTrailingLen = batchTrailing.dim(1)
        let cpSamplingConfig = SamplingConfig(temperature: sampling.temperature, topK: sampling.topK)
        let codecPadToken = Int32(CodecTokens.codecPad)

        // Prefill
        let prefillLen = batchPrefill.dim(1)

        var (logits, hiddenStates, talkerCache) = talker(
            inputsEmbeds: batchPrefill,
            offset: MLXArray(Int32(0)),
            cache: nil)

        // Sample first token for each item
        let firstLogits = logits[0..., (prefillLen - 1)..<prefillLen, 0...]  // [B, 1, vocab]
        var finished = MLXArray(Array(repeating: false, count: batchSize))  // [B]

        var nextTokens = sampleTokensBatch(
            logits: firstLogits,
            config: sampling,
            finishedMask: finished,
            padToken: codecPadToken,
            suppressRange: codecSuppressRange,
            eosTokenId: CodecTokens.codecEos)

        // Check which items hit EOS immediately
        let eosCheck = nextTokens .== MLXArray(Int32(CodecTokens.codecEos))
        finished = logicalOr(finished, eosCheck)

        // Get hidden states for code predictor
        var lastHidden = hiddenStates[0..., (prefillLen - 1)..<prefillLen, 0...]  // [B, 1, D]

        // Predict remaining 15 codebooks for first timestep
        var cpTokens = predictCodebooksForTimestepBatch(
            hiddenStates: lastHidden,
            firstCodebookTokens: nextTokens,
            cpSamplingConfig: cpSamplingConfig)  // [B, 15]

        // Accumulate codebooks: list of [B, 16] per timestep
        var allCBSteps: [MLXArray] = []
        let firstStep = concatenated([nextTokens.expandedDimensions(axis: 1), cpTokens], axis: 1)  // [B, 16]
        allCBSteps.append(firstStep)

        var trailingIdx = 0
        var step = prefillLen

        // Autoregressive generation
        for iterIdx in 1..<safeMaxTokens {
            // Text side: next trailing text embed or pad (same index for all items since pre-padded)
            let textEmbed: MLXArray
            if trailingIdx < maxTrailingLen {
                textEmbed = batchTrailing[0..., trailingIdx..<(trailingIdx + 1), 0...]  // [B, 1, D]
                trailingIdx += 1
            } else {
                // Broadcast pad embed to [B, 1, D]
                textEmbed = broadcast(ttsPadEmbed, to: [batchSize, 1, config.talker.hiddenSize])
            }

            // Codec side: embed first codebook + sum of 15 predicted codebooks
            let codecEmbed = talker.embedCodec(
                nextTokens.expandedDimensions(axis: 1))  // [B, 1] → [B, 1, D]
                + codePredictor.batchEmbedAllGroupsBatch(cpTokens)  // [B, 1, D]

            let stepEmbeds = textEmbed + codecEmbed  // [B, 1, D]

            let newResult = executeTalkerStep(
                embeds: stepEmbeds, offset: step, cache: talkerCache)
            logits = newResult.0
            hiddenStates = newResult.1
            talkerCache = newResult.2

            nextTokens = sampleTokensBatch(
                logits: logits,
                config: sampling,
                finishedMask: finished,
                padToken: codecPadToken,
                suppressRange: codecSuppressRange,
                eosTokenId: CodecTokens.codecEos)

            // Update finished mask
            let newEos = nextTokens .== MLXArray(Int32(CodecTokens.codecEos))
            finished = logicalOr(finished, newEos)

            // Code predictor for this timestep
            lastHidden = hiddenStates  // [B, 1, D]
            cpTokens = predictCodebooksForTimestepBatch(
                hiddenStates: lastHidden,
                firstCodebookTokens: nextTokens,
                cpSamplingConfig: cpSamplingConfig)

            let stepCB = concatenated([nextTokens.expandedDimensions(axis: 1), cpTokens], axis: 1)
            allCBSteps.append(stepCB)

            step += 1

            // Check if all items are done
            eval(finished)
            let finishedArray = finished.asArray(Bool.self)
            if finishedArray.allSatisfy({ $0 }) { break }

            if iterIdx % 50 == 0 {
                let estSec = Double(iterIdx) / 12.5
                let doneCount = finishedArray.filter { $0 }.count
                print("  Batch: \(iterIdx) steps (~\(String(format: "%.1f", estSec))s), \(doneCount)/\(batchSize) done...")
            }
        }

        let totalSteps = allCBSteps.count
        print("  Batch generation done: \(totalSteps) steps, \(batchSize) items")

        // Stack all timesteps: [B, 16, T]
        let stepsStacked = stacked(allCBSteps, axis: 0)  // [T, B, 16]
        let allCB = stepsStacked.transposed(1, 2, 0)  // [B, 16, T]
        eval(allCB)

        // Extract per-item codebooks, trimming at EOS
        var results: [MLXArray] = []
        var frameCounts: [Int] = []

        for i in 0..<batchSize {
            let itemCB = allCB[i..<(i + 1)]  // [1, 16, T]
            let firstCBRow = itemCB[0..., 0, 0...]  // [1, T] — first codebook
            eval(firstCBRow)
            let tokens = firstCBRow.squeezed().asArray(Int32.self)

            // Find EOS position
            var eosPos = tokens.count
            for (j, tok) in tokens.enumerated() {
                if tok == Int32(CodecTokens.codecEos) {
                    eosPos = j
                    break
                }
            }

            if eosPos == 0 {
                results.append(MLXArray.zeros([1, 16, 0]))
                frameCounts.append(0)
            } else {
                let trimmed = itemCB[0..., 0..., 0..<eosPos]  // [1, 16, eosPos]
                results.append(trimmed)
                frameCounts.append(eosPos)
            }
        }

        return (results, frameCounts)
    }

    /// Predict 15 remaining codebook tokens for B items at a single timestep.
    private func predictCodebooksForTimestepBatch(
        hiddenStates: MLXArray,
        firstCodebookTokens: MLXArray,
        cpSamplingConfig: SamplingConfig
    ) -> MLXArray {
        let batchSize = hiddenStates.dim(0)
        let numGroups = config.codePredictor.numCodeGroups - 1  // 15

        var cpCache: [(MLXArray, MLXArray)]? = nil

        // First codebook embedding (from talker's codec embedding)
        let code0Embed = talker.embedCodec(
            firstCodebookTokens.expandedDimensions(axis: 1))  // [B, 1, D]

        // Prefill: [hidden_state, code_0_embed] — length 2
        let prefillInput = concatenated([hiddenStates, code0Embed], axis: 1)  // [B, 2, D]

        // Predict codebook group 0
        var (cpLogits, cpNewCache) = codePredictor(
            inputsEmbeds: prefillInput, groupIndex: 0, cache: nil)
        cpCache = cpNewCache

        let lastCpLogits = cpLogits[0..., 1..<2, 0...]  // [B, 1, vocab]

        // No EOS/suppress needed for code predictor
        let noFinished = MLXArray(Array(repeating: false, count: batchSize))
        var prevTokens = sampleTokensBatch(
            logits: lastCpLogits,
            config: cpSamplingConfig,
            finishedMask: noFinished,
            padToken: 0)

        var groupTokens: [MLXArray] = [prevTokens]  // list of [B]

        // Remaining 14 codebook groups (compiled transformer + separate lm_head)
        for groupIdx in 1..<numGroups {
            let prevEmbed = codePredictor.embedCodecGroup(
                prevTokens.expandedDimensions(axis: 1),
                groupIndex: groupIdx - 1)  // [B, 1, D]

            let cpResult = executeCPTransformerStep(hidden: prevEmbed, cache: cpCache!)
            cpCache = cpResult.newCache
            cpLogits = codePredictor.lmHeads[groupIdx](cpResult.normed)

            prevTokens = sampleTokensBatch(
                logits: cpLogits,
                config: cpSamplingConfig,
                finishedMask: noFinished,
                padToken: 0)
            groupTokens.append(prevTokens)
        }

        // Stack: 15 × [B] → [B, 15]
        return stacked(groupTokens, axis: 1)
    }
    // MARK: - Warm-up

    /// Run minimal dummy forward passes to compile Metal shaders and allocate GPU buffers.
    /// This eliminates first-inference latency from shader compilation.
    public func warmUp() {
        guard let tokenizer = tokenizer else { return }

        // Set up compiled code predictor for kernel fusion
        setupCompilation()

        // Run a minimal prefill through the talker to compile all Metal shaders.
        let textTokens = prepareTextTokens(text: "hi", tokenizer: tokenizer)
        let warmupLangId = resolveLanguageId(for: "english") ?? CodecTokens.languageEnglish
        let codecPrefix = buildCodecPrefix(languageId: warmupLangId)
        let (prefillEmbeds, trailingTextHidden, ttsPadEmbed) = buildPrefillEmbeddings(
            textTokens: textTokens, codecPrefixTokens: codecPrefix)
        eval(prefillEmbeds, trailingTextHidden, ttsPadEmbed)

        // Talker prefill: compiles all 28-layer attention + MLP shaders
        let prefillLen = prefillEmbeds.dim(1)
        let (logits, hiddenStates, talkerWarmupCache) = talker(
            inputsEmbeds: prefillEmbeds, offset: MLXArray(Int32(0)), cache: nil)
        eval(logits)

        // Pre-compile talker generation step (shapeless=true, traced once here).
        // Uses the cache from prefill so the compiled graph includes cache concatenation.
        let warmupCodecEmbed = talker.embedCodec(MLXArray([Int32(0)]).expandedDimensions(axis: 0))
        let (warmupLogits, _, _) = executeTalkerStep(
            embeds: warmupCodecEmbed, offset: prefillLen, cache: talkerWarmupCache)
        eval(warmupLogits)

        // Parallel code predictor: compiles 5-layer shaders + all 15 lm_heads
        let lastHidden = hiddenStates[0..., (prefillLen - 1)..<prefillLen, 0...]
        let code0Embed = talker.embedCodec(MLXArray([Int32(0)]).expandedDimensions(axis: 0))
        let cpInput = concatenated([lastHidden, code0Embed], axis: 1)
        let allLogits = codePredictor.predictAllGroupsParallel(inputsEmbeds: cpInput)
        eval(allLogits)

        // Pre-compile CP transformer for all 14 cache sizes (groups 1-14).
        // Each group i has cache seqLen = i+2, which is constant across timesteps.
        // This traces + compiles 14 graphs during warmup so generation pays zero compile cost.
        let (_, cpPrefillCache) = codePredictor(
            inputsEmbeds: cpInput, groupIndex: 0, cache: nil)
        var cpCache: [(MLXArray, MLXArray)] = cpPrefillCache
        let numGroups = config.codePredictor.numCodeGroups - 1  // 15
        for groupIdx in 1..<numGroups {
            let (cpNormed, newCPCache) = executeCPTransformerStep(
                hidden: code0Embed, cache: cpCache)
            cpCache = newCPCache
            let groupLogits = codePredictor.lmHeads[groupIdx](cpNormed)
            eval(groupLogits)
        }

        // Compile codec decoder for kernel fusion (different from shader JIT compilation).
        // compile() fuses multiple kernel dispatches into fewer optimized kernels per chunk.
        // Warmup adds ~300ms to load time but saves on every generation.
        codecDecoder.setupCompilation()
        codecDecoder.warmUp()
    }

    // MARK: - Compiled Generation Steps

    /// Initialize compiled talker and code predictor for Metal kernel fusion.
    ///
    /// MLX.compile() traces the computation graph on first call and replays it
    /// on subsequent calls, fusing small kernel calls into larger ones.
    ///
    /// Talker: compiled with shapeless=true. RoPE offset is passed as a regular MLXArray
    /// input (compile treats inputs as variables, not constants). Growing KV cache is
    /// handled by shapeless mode. Batch dimension uses -1 reshapes for any batch size.
    ///
    /// Code predictor: compiled with shapeless=false. 14 fixed cache sizes, compiled
    /// once during warmup, reused for all subsequent timesteps.
    func setupCompilation() {
        // Compiled talker: [embeds, offset, K0, V0, ..., K27, V27] →
        //                  [logits, hidden, K0, V0, ..., K27, V27]
        // Offset is a regular function input (compile treats inputs as variables, not constants).
        let talkerRef = talker
        let numTalkerLayers = config.talker.numLayers

        compiledTalkerStep = compile(
            inputs: [talkerRef], outputs: [talkerRef], shapeless: true
        ) { inputs in
            let embeds = inputs[0]
            let offset = inputs[1]  // MLXArray scalar — dynamic, not baked
            var cache: [(MLXArray, MLXArray)] = []
            for i in 0..<numTalkerLayers {
                cache.append((inputs[2 + i * 2], inputs[3 + i * 2]))
            }

            let (logits, hidden, newCache) = talkerRef(
                inputsEmbeds: embeds, offset: offset, cache: cache)

            var result = [logits, hidden]
            for (k, v) in newCache { result.append(k); result.append(v) }
            return result
        }
        let numCPLayers = config.codePredictor.numLayers
        let cpRef = codePredictor

        // Compiled CP transformer: hidden + 5×(K,V) cache → normed + 5×(K,V) cache
        // Does NOT include lm_head (applied separately per group index)
        // Includes small_to_mtp_projection when talker and CP have different hidden sizes
        compiledCPTransformer = compile(
            inputs: [cpRef], outputs: [cpRef], shapeless: false
        ) { inputs in
            var hidden = inputs[0]
            // Project from talker dim to CP dim if needed (1.7B: 2048→1024)
            if let proj = cpRef.smallToMtpProjection {
                hidden = proj(hidden)
            }
            var cpCache: [(MLXArray, MLXArray)] = []
            for i in 0..<numCPLayers {
                cpCache.append((inputs[1 + i * 2], inputs[2 + i * 2]))
            }
            var newCache: [(MLXArray, MLXArray)] = []
            for (i, layer) in cpRef.layers.enumerated() {
                let (output, updated) = layer(hidden, attentionMask: nil, cache: cpCache[i])
                hidden = output
                newCache.append(updated)
            }
            hidden = cpRef.norm(hidden)
            var result = [hidden]
            for (k, v) in newCache { result.append(k); result.append(v) }
            return result
        }
    }

    /// Execute a talker generation step (compiled when available).
    ///
    /// The compiled path fuses ~420 Metal kernel dispatches (28 layers × ~15 ops) into
    /// fewer optimized kernels. Uses shapeless=true to handle growing KV cache without
    /// recompilation. RoPE offset is passed as a regular MLXArray input (not baked).
    private func executeTalkerStep(
        embeds: MLXArray, offset: Int, cache: [(MLXArray, MLXArray)]
    ) -> (MLXArray, MLXArray, [(MLXArray, MLXArray)]) {
        guard let compiled = compiledTalkerStep else {
            return talker(inputsEmbeds: embeds, offset: MLXArray(Int32(offset)), cache: cache)
        }

        // Flatten inputs: [embeds, offset, K0, V0, K1, V1, ..., K27, V27]
        let offsetArray = MLXArray(Int32(offset))
        var flatInputs = [embeds, offsetArray]
        for (k, v) in cache { flatInputs.append(k); flatInputs.append(v) }

        let out = compiled(flatInputs)

        // Unflatten: [logits, hidden, K0, V0, ..., K27, V27]
        var newCache: [(MLXArray, MLXArray)] = []
        for i in 0..<config.talker.numLayers {
            newCache.append((out[2 + i * 2], out[3 + i * 2]))
        }
        return (out[0], out[1], newCache)
    }

    /// Execute a code predictor transformer step (layers + norm, no lm_head).
    /// For single-token steps (seqLen=1) where no attention mask is needed.
    /// Apply codePredictor.lmHeads[groupIndex] to the normed output separately.
    private func executeCPTransformerStep(
        hidden: MLXArray, cache: [(MLXArray, MLXArray)]
    ) -> (normed: MLXArray, newCache: [(MLXArray, MLXArray)]) {
        guard let compiled = compiledCPTransformer else {
            var h = hidden
            // Project from talker dim to CP dim if needed (1.7B: 2048→1024)
            if let proj = codePredictor.smallToMtpProjection {
                h = proj(h)
            }
            var newCache: [(MLXArray, MLXArray)] = []
            for (i, layer) in codePredictor.layers.enumerated() {
                let (output, updated) = layer(h, attentionMask: nil, cache: cache[i])
                h = output
                newCache.append(updated)
            }
            return (codePredictor.norm(h), newCache)
        }
        var flatInputs = [hidden]
        for (k, v) in cache { flatInputs.append(k); flatInputs.append(v) }
        let out = compiled(flatInputs)
        var newCache: [(MLXArray, MLXArray)] = []
        for i in 0..<config.codePredictor.numLayers {
            newCache.append((out[1 + i * 2], out[2 + i * 2]))
        }
        return (out[0], newCache)
    }

    // MARK: - Speaker Resolution

    /// Resolve speaker name to token ID and effective language.
    /// - Returns: (speakerTokenId, effectiveLanguage) — speakerTokenId is nil if no speaker
    private func resolveSpeaker(_ speaker: String?, language: String) -> (Int?, String) {
        guard let speakerName = speaker else {
            return (nil, language)
        }

        guard let config = speakerConfig else {
            print("Warning: Speaker '\(speakerName)' requested but model has no speaker support. " +
                  "Use the CustomVoice model variant for speaker selection.")
            return (nil, language)
        }

        let normalizedName = speakerName.lowercased()
        guard let tokenId = config.speakerIds[normalizedName] else {
            let available = config.availableSpeakers.joined(separator: ", ")
            print("Warning: Unknown speaker '\(speakerName)'. Available speakers: \(available)")
            return (nil, language)
        }

        // Check if this speaker has a dialect override
        var effectiveLanguage = language
        if let dialect = config.speakerDialects[normalizedName] {
            effectiveLanguage = dialect
        }

        return (tokenId, effectiveLanguage)
    }

    /// Resolve language name to codec language token ID.
    /// Checks config-based language map first (from config.json), falls back to static CodecTokens.
    private func resolveLanguageId(for language: String) -> Int? {
        let normalized = language.lowercased()
        // Config-based map (includes all languages for this model variant)
        if let langMap = config.talker.codecLanguageIds, let id = langMap[normalized] {
            return id
        }
        // Fallback to static constants (handles short codes like "en", "zh")
        return CodecTokens.languageId(for: normalized)
    }

    // MARK: - Text Preparation

    /// Prepare instruction tokens for CustomVoice instruct mode.
    /// Format: `<|im_start|>user\n{instruct}<|im_end|>\n`
    ///
    /// The instruct tokens are embedded via `text_embedding → text_projection` and prepended
    /// before the existing prefill sequence (role + codec overlay + first_text).
    func prepareInstructTokens(instruct: String, tokenizer: Qwen3Tokenizer) -> [Int] {
        let imStartId = 151644
        let imEndId = 151645
        let newlineId = 198
        let userId = 872

        var tokens: [Int] = [imStartId, userId, newlineId]
        tokens.append(contentsOf: tokenizer.encode(instruct))
        tokens.append(contentsOf: [imEndId, newlineId])
        return tokens
    }

    /// Prepare text tokens using chat template.
    /// Template: <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
    private func prepareTextTokens(text: String, tokenizer: Qwen3Tokenizer) -> [Int] {
        let imStartId = 151644
        let imEndId = 151645
        let newlineId = 198
        let assistantId = 77091

        var tokens: [Int] = []

        // <|im_start|>assistant\n
        tokens.append(contentsOf: [imStartId, assistantId, newlineId])

        // Encode text
        let textTokens = tokenizer.encode(text)
        tokens.append(contentsOf: textTokens)

        // <|im_end|>\n<|im_start|>assistant\n
        tokens.append(contentsOf: [imEndId, newlineId, imStartId, assistantId, newlineId])

        return tokens
    }

    // MARK: - Codec Prefix

    /// Build codec prefix: [think, think_bos, lang_id, think_eos, pad, bos] (6 tokens)
    /// With speaker: [think, think_bos, lang_id, think_eos, spk_token, pad, bos] (7 tokens)
    ///
    /// Python reference inserts speaker between think_eos and pad/bos suffix:
    ///   codec_prefill = [think, think_bos, lang, think_eos]
    ///   codec_suffix  = [pad, bos]
    ///   codec_embed   = concat([codec_prefill, speaker_embed, codec_suffix])
    /// The speaker token must precede pad+bos so that the text overlay alignment
    /// maps tts_bos to the pad position and first_text to codec_bos (the last token).
    func buildCodecPrefix(languageId: Int, speakerTokenId: Int? = nil) -> [Int32] {
        let tc = config.talker
        var prefix: [Int32] = [
            Int32(tc.codecThinkId),
            Int32(tc.codecThinkBosId),
            Int32(languageId),
            Int32(tc.codecThinkEosId),
        ]
        if let spkId = speakerTokenId {
            prefix.append(Int32(spkId))
        }
        prefix.append(contentsOf: [
            Int32(tc.codecPadId),
            Int32(tc.codecBosId),
        ])
        return prefix
    }

    // MARK: - Embedding Construction

    /// Build prefill embeddings with element-wise text+codec overlay.
    ///
    /// Python reference:
    /// ```
    /// # Text-side TTS special tokens
    /// tts_bos_embed = text_projection(text_embedding(151672))
    /// tts_pad_embed = text_projection(text_embedding(151671))
    /// tts_eos_embed = text_projection(text_embedding(151673))
    ///
    /// # Build text overlay for codec prefix (pad_count = codec_len - 2)
    /// pad_embeds = broadcast(tts_pad_embed, (1, pad_count, 1024))
    /// text_overlay = concat([pad_embeds, tts_bos_embed], axis=1)
    ///
    /// # Element-wise sum of text overlay + codec prefix (minus last token)
    /// combined = text_overlay + codec_embed[:, :-1, :]
    ///
    /// # Role embedding (first 3 text tokens: <|im_start|>assistant\n)
    /// role_embed = text_embed[:, :3, :]
    ///
    /// # First text token added to last codec token (codec_bos)
    /// first_text = text_embed[:, 3:4, :] + codec_embed[:, -1:, :]
    ///
    /// # Prefill input
    /// input_embeds = concat([role_embed, combined, first_text], axis=1)
    ///
    /// # Trailing text (tokens 4 to -5, plus tts_eos)
    /// trailing_text = concat([text_embed[:, 4:-5, :], tts_eos_embed], axis=1)
    /// ```
    /// Build prefill embeddings for the Talker model.
    ///
    /// Two modes are supported, matching the Python reference (`modeling_qwen3_tts.py` lines 2198-2232):
    ///
    /// **Streaming mode** (`nonStreamingMode = false`):
    /// Packs only the first text token into the prefill. Remaining text tokens are returned as
    /// `trailingTextHidden` and fed one-per-codec-step during autoregressive generation.
    /// Used by voice cloning and streaming synthesis.
    ///
    /// **Non-streaming mode** (`nonStreamingMode = true`):
    /// Packs ALL text tokens into the prefill, each summed with a `codec_pad` embedding.
    /// The model sees every word before generating any audio. `trailingTextHidden` is just
    /// `tts_pad_embed` (no real text remains). Used by VoiceDesign synthesis — the Python
    /// `generate_voice_design()` defaults to `non_streaming_mode=True`.
    ///
    /// **IMPORTANT — Do NOT remove non-streaming mode.**
    /// Without it, multi-sentence text produces silence after the first sentence (~4.2s)
    /// because the trailing text pool (one token per codec step) exhausts long before the
    /// model finishes generating audio. The model generates ~4.4x more codec tokens than
    /// text tokens, so streaming mode runs out of text early and switches to tts_pad,
    /// causing the model to output silence for the remaining sentences.
    private func buildPrefillEmbeddings(
        textTokens: [Int], codecPrefixTokens: [Int32], instructTokens: [Int]? = nil,
        speakerEmbedding: MLXArray? = nil,
        nonStreamingMode: Bool = false
    ) -> (prefillEmbeds: MLXArray, trailingTextHidden: MLXArray, ttsPadEmbed: MLXArray) {
        let hiddenSize = config.talker.hiddenSize

        // Embed all text tokens → project to hidden dim
        let textTokenArray = MLXArray(textTokens.map { Int32($0) }).expandedDimensions(axis: 0)  // [1, textLen]
        let textEmbeds = talker.embedText(textTokenArray)  // [1, textLen, hiddenSize]

        // Embed codec prefix tokens
        let codecArray = MLXArray(codecPrefixTokens).expandedDimensions(axis: 0)  // [1, codecLen]
        var codecEmbeds = talker.embedCodec(codecArray)  // [1, codecLen, hiddenSize]

        // Voice cloning: inject speaker embedding between think tokens and pad/bos
        // Python: cat([codec_embed[:,:4,:], speaker_embed.view(1,1,-1), codec_embed[:,4:,:]])
        if let spkEmbed = speakerEmbedding {
            let spkEmbedReshaped = spkEmbed.reshaped([1, 1, hiddenSize])  // [1, 1, 1024]
            let part0 = codecEmbeds[0..., 0..<4, 0...]  // [think, think_bos, lang, think_eos]
            let part1 = codecEmbeds[0..., 4..., 0...]    // [pad, bos]
            codecEmbeds = concatenated([part0, spkEmbedReshaped, part1], axis: 1)  // [1, 7, 1024]
        }

        // TTS special token embeddings (text-side)
        let ttsPadTokens = MLXArray([Int32(CodecTokens.ttsPad)]).expandedDimensions(axis: 0)
        let ttsBosTokens = MLXArray([Int32(CodecTokens.ttsBos)]).expandedDimensions(axis: 0)
        let ttsEosTokens = MLXArray([Int32(CodecTokens.ttsEos)]).expandedDimensions(axis: 0)

        let ttsPadEmbed = talker.embedText(ttsPadTokens)  // [1, 1, hiddenSize]
        let ttsBosEmbed = talker.embedText(ttsBosTokens)  // [1, 1, hiddenSize]
        let ttsEosEmbed = talker.embedText(ttsEosTokens)  // [1, 1, hiddenSize]

        let codecLen = codecEmbeds.dim(1)  // 6 without speaker, 7 with speaker

        // Text overlay for codec prefix:
        // pad_count = codecLen - 2 (the last 2 positions are: tts_bos overlay + first_text+codec_bos)
        // But actually: we overlay codecLen-1 positions with text, and the last codec token (bos) gets first_text
        let padCount = codecLen - 2  // 4 pad positions
        let padEmbeds = broadcast(ttsPadEmbed, to: [1, padCount, hiddenSize])
        let textOverlay = concatenated([padEmbeds, ttsBosEmbed], axis: 1)  // [1, codecLen-1, hiddenSize]

        // Element-wise sum: text overlay + codec prefix (all but last token)
        let codecWithoutLast = codecEmbeds[0..., 0..<(codecLen - 1), 0...]  // [1, codecLen-1, hiddenSize]
        let combined = textOverlay + codecWithoutLast  // [1, codecLen-1, hiddenSize]

        // Role embedding: first 3 text tokens (<|im_start|>assistant\n)
        let roleEmbed = textEmbeds[0..., 0..<3, 0...]  // [1, 3, hiddenSize]

        // textTokens layout: [im_start, assistant, \n, ...text..., im_end, \n, im_start, assistant, \n]
        // - Index 0-2: role header (im_start, assistant, \n) → roleEmbed
        // - Index 3 to (len-6): actual text tokens
        // - Last 5: suffix (im_end, \n, im_start, assistant, \n) → excluded

        let textLen = textTokens.count

        if nonStreamingMode {
            // ── Non-streaming mode (Python lines 2203-2227) ──
            // Pack ALL text tokens into the prefill. Each text token is summed with a codec_pad
            // embedding, and a final tts_pad + codec_bos caps the sequence. The model sees the
            // entire text before generating any codec tokens.
            //
            // Streaming prefill would end with: [..., first_text + codec_bos]
            // Non-streaming replaces that last position with:
            //   [all_text + codec_pad, ..., all_text + codec_pad, tts_eos + codec_pad, tts_pad + codec_bos]

            // Base prefill without the last position (first_text + codec_bos)
            let prefillBase: MLXArray
            if let instructTokens = instructTokens {
                let instructArray = MLXArray(instructTokens.map { Int32($0) }).expandedDimensions(axis: 0)
                let instructEmbeds = talker.embedText(instructArray)  // [1, N, hiddenSize]
                prefillBase = concatenated([instructEmbeds, roleEmbed, combined], axis: 1)
            } else {
                prefillBase = concatenated([roleEmbed, combined], axis: 1)
            }

            // All text tokens [3:-5] + tts_eos, each paired with codec_pad
            let allTextStart = 3
            let allTextEnd = textLen - 5
            let numTextTokens = allTextEnd - allTextStart
            if numTextTokens > 0 {
                let allTextEmbeds = textEmbeds[0..., allTextStart..<allTextEnd, 0...]  // [1, N, D]
                let textWithEos = concatenated([allTextEmbeds, ttsEosEmbed], axis: 1)  // [1, N+1, D]

                // codec_pad embeddings for each text+eos position
                let codecPadCount = textWithEos.dim(1)
                let codecPadIds = MLXArray(Array(repeating: Int32(CodecTokens.codecPad), count: codecPadCount))
                    .expandedDimensions(axis: 0)  // [1, N+1]
                let codecPadEmbeds = talker.embedCodec(codecPadIds)  // [1, N+1, D]
                let textCodecCombined = textWithEos + codecPadEmbeds

                // Final position: tts_pad + codec_bos
                let codecBosIds = MLXArray([Int32(CodecTokens.codecBos)]).expandedDimensions(axis: 0)
                let codecBosEmbed = talker.embedCodec(codecBosIds)  // [1, 1, D]
                let finalPos = ttsPadEmbed + codecBosEmbed

                let prefillEmbeds = concatenated([prefillBase, textCodecCombined, finalPos], axis: 1)
                // All text is in prefill — trailing is just tts_pad (no real text to feed)
                return (prefillEmbeds, ttsPadEmbed, ttsPadEmbed)
            } else {
                // Very short text: fall through to streaming-like behavior
                let firstTextEmbed = textEmbeds[0..., 3..<4, 0...]
                let lastCodecEmbed = codecEmbeds[0..., (codecLen - 1)..<codecLen, 0...]
                let firstTextPlusCodec = firstTextEmbed + lastCodecEmbed
                let prefillEmbeds = concatenated([prefillBase, firstTextPlusCodec], axis: 1)
                return (prefillEmbeds, ttsEosEmbed, ttsPadEmbed)
            }
        } else {
            // ── Streaming mode (Python lines 2228-2232) ──
            // Only the first text token goes into the prefill. Remaining text tokens are
            // returned as trailingTextHidden, fed one-per-codec-step during generation.
            // Used by voice cloning (Python `generate()` defaults non_streaming_mode=False).

            // First text token (index 3) added to last codec token (codec_bos)
            let firstTextEmbed = textEmbeds[0..., 3..<4, 0...]  // [1, 1, hiddenSize]
            let lastCodecEmbed = codecEmbeds[0..., (codecLen - 1)..<codecLen, 0...]  // [1, 1, hiddenSize]
            let firstTextPlusCodec = firstTextEmbed + lastCodecEmbed  // [1, 1, hiddenSize]

            // Prefill: [instruct? | role_embed, combined, first_text+codec_bos]
            let prefillEmbeds: MLXArray
            if let instructTokens = instructTokens {
                let instructArray = MLXArray(instructTokens.map { Int32($0) }).expandedDimensions(axis: 0)
                let instructEmbeds = talker.embedText(instructArray)  // [1, N, hiddenSize]
                prefillEmbeds = concatenated([instructEmbeds, roleEmbed, combined, firstTextPlusCodec], axis: 1)
            } else {
                prefillEmbeds = concatenated([roleEmbed, combined, firstTextPlusCodec], axis: 1)
            }

            // Trailing text: tokens[4:-5] + tts_eos
            let trailStart = 4
            let trailEnd = textLen - 5
            if trailEnd > trailStart {
                let trailingSlice = textEmbeds[0..., trailStart..<trailEnd, 0...]
                let trailingTextHidden = concatenated([trailingSlice, ttsEosEmbed], axis: 1)
                return (prefillEmbeds, trailingTextHidden, ttsPadEmbed)
            } else {
                // Very short text: only tts_eos as trailing
                return (prefillEmbeds, ttsEosEmbed, ttsPadEmbed)
            }
        }
    }

    // MARK: - Generation Loop

    /// Generate first codebook + predict remaining 15 codebooks per-step.
    ///
    /// Each generation step:
    /// 1. Run talker forward → get logits for first codebook + hidden states
    /// 2. Sample first codebook token
    /// 3. Run code predictor autoregressively to predict 15 remaining codebook tokens
    /// 4. Build next-step input: trailing_text_embed + sum(all 16 codebook embeddings)
    private func generateWithCodePredictor(
        prefillEmbeds: MLXArray,
        trailingTextHidden: MLXArray,
        ttsPadEmbed: MLXArray,
        sampling: SamplingConfig
    ) -> (allCodebooks: MLXArray, numFrames: Int) {
        // Seed RNG once at the start of generation (not per-token).
        // Per-token seeding caused runaway generation for certain seed values.
        if let seed = sampling.seed {
            seedSamplingRNG(seed)
        }

        let safeMaxTokens = min(sampling.maxTokens, 1500)

        var talkerCache: [(MLXArray, MLXArray)]? = nil
        var generatedFirstCodebook: [Int32] = []
        var generatedAllCodebooks: [[Int32]] = (0..<config.codePredictor.numCodeGroups).map { _ in [] }

        // Pre-allocate code predictor sampling config (reused every step × 15 groups)
        let cpSamplingConfig = SamplingConfig(temperature: sampling.temperature, topK: sampling.topK)

        let prefillLen = prefillEmbeds.dim(1)

        // Prefill
        var (logits, hiddenStates, newCache) = talker(
            inputsEmbeds: prefillEmbeds,
            offset: MLXArray(Int32(0)),
            cache: talkerCache)
        talkerCache = newCache

        // Sample first token from last position
        let lastLogits = logits[0..., (prefillLen - 1)..<prefillLen, 0...]
        var nextToken = sampleToken(
            logits: lastLogits,
            config: sampling,
            generatedTokens: generatedFirstCodebook,
            suppressRange: codecSuppressRange,
            eosTokenId: CodecTokens.codecEos)

        if nextToken == Int32(CodecTokens.codecEos) {
            return (MLXArray.zeros([1, 16, 0]), 0)
        }

        generatedFirstCodebook.append(nextToken)
        generatedAllCodebooks[0].append(nextToken)

        // Get hidden state for this step's code predictor
        let lastHidden = hiddenStates[0..., (prefillLen - 1)..<prefillLen, 0...]  // [1, 1, D]

        // Run code predictor for this timestep to get remaining 15 codebook tokens
        var codeTokens = predictCodebooksForTimestep(
            hiddenState: lastHidden,
            firstCodebookToken: nextToken,
            cpSamplingConfig: cpSamplingConfig)
        for (i, token) in codeTokens.enumerated() {
            generatedAllCodebooks[i + 1].append(token)
        }

        var trailingIdx = 0
        var step = prefillLen

        // Autoregressive generation
        for iterIdx in 1..<safeMaxTokens {
            // Text side: next trailing text embed or tts_pad
            let textEmbed: MLXArray
            let trailingLen = trailingTextHidden.dim(1)
            if trailingIdx < trailingLen {
                textEmbed = trailingTextHidden[0..., trailingIdx..<(trailingIdx + 1), 0...]
                trailingIdx += 1
            } else {
                textEmbed = ttsPadEmbed
            }

            // Codec side: sum all 16 codebook embeddings (codebook 0 + 15 code predictor groups)
            let codecEmbed = talker.embedCodec(
                MLXArray([nextToken]).expandedDimensions(axis: 0))  // [1, 1, D] — codebook 0
                + codePredictor.batchEmbedAllGroups(codeTokens)     // [1, 1, D] — groups 1-15

            // Next input = text + codec (element-wise sum)
            let stepEmbeds = textEmbed + codecEmbed  // [1, 1, D]

            (logits, hiddenStates, newCache) = executeTalkerStep(
                embeds: stepEmbeds, offset: step, cache: talkerCache!)
            talkerCache = newCache

            nextToken = sampleToken(
                logits: logits,
                config: sampling,
                generatedTokens: generatedFirstCodebook,
                suppressRange: codecSuppressRange,
                eosTokenId: CodecTokens.codecEos)

            if nextToken == Int32(CodecTokens.codecEos) { break }

            generatedFirstCodebook.append(nextToken)
            generatedAllCodebooks[0].append(nextToken)

            // Code predictor for this timestep
            let stepHidden = hiddenStates  // [1, 1, D]
            codeTokens = predictCodebooksForTimestep(
                hiddenState: stepHidden,
                firstCodebookToken: nextToken,
                cpSamplingConfig: cpSamplingConfig)
            for (i, token) in codeTokens.enumerated() {
                generatedAllCodebooks[i + 1].append(token)
            }

            step += 1

            if iterIdx % 50 == 0 {
                let estSec = Double(generatedFirstCodebook.count) / 12.5
                print("  Talker: \(generatedFirstCodebook.count) tokens (~\(String(format: "%.1f", estSec))s audio)...")
            }
        }

        let numFrames = generatedFirstCodebook.count

        if numFrames >= safeMaxTokens && nextToken != Int32(CodecTokens.codecEos) {
            let estSec = Double(numFrames) / 12.5
            print("Warning: Hit safety limit of \(safeMaxTokens) tokens (~\(String(format: "%.1f", estSec))s audio). "
                + "Increase SamplingConfig.maxTokens if you need longer output.")
        }

        let estAudioSec = Double(numFrames) / 12.5
        print("  Talker done: \(numFrames) codec tokens (~\(String(format: "%.1f", estAudioSec))s audio)")

        // Stack all codebooks: [1, 16, T]
        let codebookArrays = generatedAllCodebooks.map { tokens in
            MLXArray(tokens).expandedDimensions(axis: 0)  // [1, T]
        }
        let allCodebooks = stacked(codebookArrays, axis: 1)  // [1, 16, T]

        return (allCodebooks, numFrames)
    }

    // MARK: - Per-Timestep Code Prediction

    /// Predict 15 remaining codebook tokens for a single timestep.
    ///
    /// The code predictor runs autoregressively across codebook groups:
    /// - Step 0: prefill [hidden_state, code_0_embed] (length 2)
    /// - Steps 1-14: single embedding of previous code token (length 1), KV cache
    ///
    /// Uses lazy evaluation: all 15 groups are chained as a single MLX computation graph
    /// with zero GPU sync barriers. One `eval()` at the end materializes all tokens.
    /// This reduces per-step GPU syncs from 15 to 1.
    private func predictCodebooksForTimestep(
        hiddenState: MLXArray,
        firstCodebookToken: Int32,
        cpSamplingConfig: SamplingConfig
    ) -> [Int32] {
        var cpCache: [(MLXArray, MLXArray)]? = nil

        // First codebook embedding (from talker's codec embedding)
        let code0Embed = talker.embedCodec(
            MLXArray([firstCodebookToken]).expandedDimensions(axis: 0))  // [1, 1, D]

        // Prefill: [hidden_state, code_0_embed] — length 2
        let prefillInput = concatenated([hiddenState, code0Embed], axis: 1)  // [1, 2, D]

        // Predict codebook group 0 (= codebook 2 overall)
        let (cpLogits, cpNewCache) = codePredictor(
            inputsEmbeds: prefillInput, groupIndex: 0, cache: nil)
        cpCache = cpNewCache

        // Sample lazily — returns MLXArray scalar, NO .item() sync
        let lastCpLogits = cpLogits[0..., 1..<2, 0...]
        var prevTokenArray = sampleTokenLazy(logits: lastCpLogits, config: cpSamplingConfig)
        var lazyTokens: [MLXArray] = [prevTokenArray]

        // Remaining 14 codebook groups — fully lazy chain, no GPU syncs
        for groupIdx in 1..<(config.codePredictor.numCodeGroups - 1) {
            // Embed previous group's token (lazy MLXArray → embedding, no sync needed)
            let prevEmbed = codePredictor.embedCodecGroup(
                prevTokenArray.reshaped(1, 1),
                groupIndex: groupIdx - 1)  // [1, 1, D]

            let cpResult = executeCPTransformerStep(hidden: prevEmbed, cache: cpCache!)
            cpCache = cpResult.newCache
            let groupLogits = codePredictor.lmHeads[groupIdx](cpResult.normed)

            prevTokenArray = sampleTokenLazy(logits: groupLogits, config: cpSamplingConfig)
            lazyTokens.append(prevTokenArray)
        }

        // ONE eval to materialize the entire 15-group computation graph
        let tokenStack = stacked(lazyTokens)  // [15]
        eval(tokenStack)
        return tokenStack.asArray(Int32.self)  // bulk extraction, no per-token sync
    }

    // MARK: - Audio Post-Processing

    /// Trim tonal tail artifacts from generated audio.
    ///
    /// **Why this is needed:** MLX's `scaledDotProductAttention` produces subtly different
    /// logit distributions compared to the CUDA flash-attn3 kernel used by the HuggingFace
    /// Spaces demo. This causes our Talker to generate extra codec tokens past the natural
    /// end of speech. Those extra tokens decode to narrow-band tonal hum (~1200 Hz) rather
    /// than silence. The demo produces clean audio because flash-attn3's different numerics
    /// lead to earlier natural EOS emission.
    ///
    /// **Approach:** Scan backward through the audio in 50ms windows. Real speech has
    /// variable energy across consecutive windows, while tonal artifacts have very consistent
    /// (low-variance) energy. We detect the transition from speech to tonal artifact and
    /// apply a fade-out at that point.
    ///
    /// - Parameters:
    ///   - samples: Raw PCM float32 samples at 24kHz
    ///   - sampleRate: Sample rate (expected 24000)
    /// - Returns: Trimmed audio with fade-out applied, or original if no tonal tail detected
    func trimTonalTail(_ samples: [Float], sampleRate: Int = 24000) -> [Float] {
        let windowSamples = Int(Double(sampleRate) * 0.05)  // 50ms windows
        let hopSamples = windowSamples / 2                   // 25ms hop (50% overlap)
        let minWindows = 4                                    // Need at least 4 windows to detect
        let fadeMs: Double = 100                              // Fade-out duration in ms

        guard samples.count > windowSamples * minWindows else { return samples }

        // Compute per-window RMS energy
        var windowEnergies: [(startSample: Int, rms: Float)] = []
        var pos = 0
        while pos + windowSamples <= samples.count {
            var sumSq: Float = 0
            for i in pos..<(pos + windowSamples) {
                sumSq += samples[i] * samples[i]
            }
            let rms = sqrt(sumSq / Float(windowSamples))
            windowEnergies.append((startSample: pos, rms: rms))
            pos += hopSamples
        }

        guard windowEnergies.count >= minWindows else { return samples }

        // Detect tonal tail: scan backward looking for a region where energy variance is very low
        // (tonal = steady hum) followed by a region with higher variance (speech).
        // We use a sliding analysis window of 6 RMS values and check variance.
        let analysisLen = 6  // ~150ms of context for variance computation
        let varianceThreshold: Float = 0.0002  // Low variance = tonal (steady energy)
        let minTonalRms: Float = 0.005  // Must have some energy to be tonal (not silence)

        var tonalStartWindowIdx: Int? = nil

        // Scan forward to find where tonal region starts
        if windowEnergies.count >= analysisLen {
            for startIdx in stride(from: windowEnergies.count - analysisLen, through: 0, by: -1) {
                let slice = windowEnergies[startIdx..<(startIdx + analysisLen)]
                let rmsValues = slice.map { $0.rms }
                let mean = rmsValues.reduce(0, +) / Float(rmsValues.count)

                // Skip if no energy (silence, not tonal)
                guard mean > minTonalRms else { continue }

                // Compute variance
                let variance = rmsValues.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(rmsValues.count)

                if variance < varianceThreshold {
                    // This region is tonal — record it but keep scanning backward
                    tonalStartWindowIdx = startIdx
                } else {
                    // Hit a region with variable energy (speech) — stop
                    if tonalStartWindowIdx != nil {
                        break
                    }
                }
            }
        }

        guard let tonalStart = tonalStartWindowIdx else {
            // No tonal tail detected
            return samples
        }

        // The trim point is the start of the tonal region, plus a 0.25s safety buffer
        // to avoid cutting off final speech that may bleed into the tonal detection zone.
        let safetyBufferSamples = Int(Double(sampleRate) * 0.25)  // 0.25 seconds
        let rawTrimSample = windowEnergies[tonalStart].startSample
        let trimSample = min(rawTrimSample + safetyBufferSamples, samples.count)

        // Don't trim if it would remove more than 50% of the audio (safety check)
        guard trimSample > samples.count / 2 else { return samples }

        // Apply fade-out
        let fadeSamples = Int(Double(sampleRate) * fadeMs / 1000.0)
        let fadeStart = max(0, trimSample - fadeSamples / 2)
        let fadeEnd = min(samples.count, trimSample + fadeSamples / 2)

        var result = Array(samples[0..<fadeEnd])
        for i in fadeStart..<fadeEnd {
            let progress = Float(i - fadeStart) / Float(fadeEnd - fadeStart)
            result[i] *= (1.0 - progress)
        }

        let trimmedDur = String(format: "%.2f", Double(result.count) / Double(sampleRate))
        let originalDur = String(format: "%.2f", Double(samples.count) / Double(sampleRate))
        print("  Tail trim: \(originalDur)s -> \(trimmedDur)s (removed \(String(format: "%.2f", Double(samples.count - result.count) / Double(sampleRate)))s tonal tail)")

        return result
    }
}

// MARK: - Model Loading

public extension Qwen3TTSModel {
    /// Load model from HuggingFace hub or a local directory.
    ///
    /// If `modelId` is a path to an existing directory on disk (absolute or relative),
    /// it is used directly and no download is attempted.
    static func fromPretrained(
        modelId: String = "aufklarer/Qwen3-TTS-12Hz-0.6B-Base-MLX-4bit",
        tokenizerModelId: String = "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> Qwen3TTSModel {
        progressHandler?(0.05, "Preparing download...")

        // Resolve main model directory — local path or HuggingFace cache
        let mainCacheDir: URL
        let localURL = URL(fileURLWithPath: modelId)
        if FileManager.default.fileExists(atPath: localURL.appendingPathComponent("config.json").path) {
            // Local directory with config.json — use directly
            mainCacheDir = localURL
            progressHandler?(0.4, "Using local model at \(modelId)")
        } else {
            // HuggingFace model ID — download if needed
            mainCacheDir = try HuggingFaceDownloader.getCacheDirectory(for: modelId)
            if !HuggingFaceDownloader.weightsExist(in: mainCacheDir) {
                progressHandler?(0.1, "Downloading TTS model weights...")
                try await HuggingFaceDownloader.downloadWeights(
                    modelId: modelId,
                    to: mainCacheDir,
                    additionalFiles: ["vocab.json", "merges.txt", "tokenizer_config.json"],
                    progressHandler: { progress in
                        progressHandler?(0.1 + progress * 0.3, "Downloading TTS model...")
                    })
            }
        }

        // Resolve tokenizer/codec directory — local path or HuggingFace cache
        let tokenizerCacheDir: URL
        let localTokenizerURL = URL(fileURLWithPath: tokenizerModelId)
        if HuggingFaceDownloader.weightsExist(in: localTokenizerURL) {
            tokenizerCacheDir = localTokenizerURL
            progressHandler?(0.4, "Using local speech tokenizer at \(tokenizerModelId)")
        } else {
            tokenizerCacheDir = try HuggingFaceDownloader.getCacheDirectory(for: tokenizerModelId)
            if !HuggingFaceDownloader.weightsExist(in: tokenizerCacheDir) {
                progressHandler?(0.4, "Downloading speech tokenizer...")
                try await HuggingFaceDownloader.downloadWeights(
                    modelId: tokenizerModelId,
                    to: tokenizerCacheDir,
                    progressHandler: { progress in
                        progressHandler?(0.4 + progress * 0.2, "Downloading speech tokenizer...")
                    })
            }
        }

        // Load config dynamically from config.json (dimensions, model type, tokens, speakers)
        let config: Qwen3TTSConfig
        let configPath = mainCacheDir.appendingPathComponent("config.json")
        if FileManager.default.fileExists(atPath: configPath.path) {
            config = try Qwen3TTSConfig.fromConfigJSON(at: mainCacheDir)
            print("  Loaded config: \(config.ttsModelType) (\(config.ttsModelSize)), " +
                  "talker=\(config.talker.hiddenSize)d, codePredictor=\(config.codePredictor.hiddenSize)d")
        } else {
            config = .base06B  // fallback to hardcoded 0.6B defaults
        }

        let model = Qwen3TTSModel(config: config)

        // Populate speaker config from parsed talker config fields
        if let spkIds = config.talker.speakerIds, !spkIds.isEmpty {
            model.speakerConfig = SpeakerConfig(
                speakerIds: spkIds,
                speakerDialects: config.talker.speakerDialects ?? [:],
                codecLanguageIds: config.talker.codecLanguageIds ?? [:])
        }

        // Load tokenizer
        progressHandler?(0.6, "Loading tokenizer...")
        let vocabPath = mainCacheDir.appendingPathComponent("vocab.json")
        if FileManager.default.fileExists(atPath: vocabPath.path) {
            let tokenizer = Qwen3Tokenizer()
            try tokenizer.load(from: vocabPath)
            model.setTokenizer(tokenizer)
        }

        // Load Talker + Code Predictor weights
        progressHandler?(0.7, "Loading TTS model weights...")
        try TTSWeightLoader.loadTalkerAndCodePredictorWeights(
            talker: model.talker, codePredictor: model.codePredictor, from: mainCacheDir)

        // Load Speaker Encoder weights (Base model only — CustomVoice/VoiceDesign don't have them)
        do {
            try TTSWeightLoader.loadSpeakerEncoderWeights(
                into: model.speakerEncoder, from: mainCacheDir)
        } catch {
            if config.ttsModelType == "voice_design" || config.ttsModelType == "custom_voice" {
                print("  Speaker encoder not found (\(config.ttsModelType) model) — OK")
            } else {
                throw error  // Base model requires speaker encoder for voice cloning
            }
        }

        // Load Speech Tokenizer Decoder weights
        progressHandler?(0.85, "Loading speech tokenizer decoder...")
        try TTSWeightLoader.loadSpeechTokenizerDecoderWeights(
            into: model.codecDecoder, from: tokenizerCacheDir)

        // Load Speech Tokenizer Encoder weights (Base model only — for voice cloning)
        // Encoder weights live in the tokenizer directory alongside decoder weights.
        let encoderKeyExists = (try? CommonWeightLoader.loadAllSafetensors(from: tokenizerCacheDir))?
            .keys.contains(where: { $0.hasPrefix("encoder.") }) ?? false
        if encoderKeyExists {
            progressHandler?(0.9, "Loading speech tokenizer encoder...")
            let encoderConfig = config.speechTokenizerEncoder ?? SpeechTokenizerEncoderConfig()
            let encoder = SpeechTokenizerEncoder(config: encoderConfig)
            try TTSWeightLoader.loadSpeechTokenizerEncoderWeights(
                into: encoder, from: tokenizerCacheDir)
            model.codecEncoder = encoder
            print("  Speech tokenizer encoder loaded (for voice cloning)")
        }

        progressHandler?(0.95, "Warming up model...")
        model.warmUp()

        progressHandler?(1.0, "Ready")
        return model
    }

}
