import Foundation
import MLX
import MLXNN
import Qwen3TTS
import AudioCommon

/// Main orchestrator for ICL voice cloning with Qwen3-TTS.
///
/// Coordinates the full voice cloning pipeline:
/// 1. **Prompt creation** — Extract speaker embedding, encode reference audio, tokenize reference text with ChatML role
/// 2. **Synthesis** — Build ICL prefill (non-streaming), run autoregressive generation, vocoder decode with reference prepending
///
/// ## CRITICAL FIX: ChatML Role Wrapping
///
/// The Python `generate_voice_clone()` wraps the reference transcript in a `ref_text`
/// ChatML role before constructing the ICL prefill. Without this, the model sees both
/// ref_text and target_text as undifferentiated "assistant" speech and regenerates the
/// reference text words before the target text — the exact "bleed" symptom.
///
/// The conversation template for ICL is:
/// ```
/// <|im_start|>ref_text\n{ref_transcript}<|im_end|>\n
/// <|im_start|>assistant\n{target_text}<|im_end|>\n
/// <|im_start|>assistant\n
/// ```
///
/// ## Usage
///
/// ```swift
/// let engine = VoiceCloningEngine(model: ttsModel, tokenizer: tokenizer)
///
/// let prompt = try engine.createPrompt(
///     referenceAudio: audioSamples,
///     referenceText: "The reference transcript.",
///     sampleRate: 24000,
///     language: "english"
/// )
///
/// let audio = try engine.synthesize(
///     text: "Hello, this is the cloned voice speaking.",
///     prompt: prompt,
///     mode: .icl
/// )
/// ```
public final class VoiceCloningEngine: @unchecked Sendable {

    // --- Model references ---
    private let talker: TalkerModel
    private let codePredictor: CodePredictorModel
    private let codecDecoder: SpeechTokenizerDecoder
    private let speakerEncoder: SpeakerEncoder
    private let config: Qwen3TTSConfig
    private let tokenizer: Qwen3Tokenizer

    // --- Native codec encoder (matched codebooks with decoder) ---
    private let referenceEncoder: ReferenceEncoder

    /// Initialize from a loaded Qwen3TTSModel and tokenizer.
    public init(model: Qwen3TTSModel, tokenizer: Qwen3Tokenizer) {
        self.talker = model.talker
        self.codePredictor = model.codePredictor
        self.codecDecoder = model.codecDecoder
        self.speakerEncoder = model.speakerEncoder
        self.config = model.config
        self.tokenizer = tokenizer
        self.referenceEncoder = ReferenceEncoder()

        // Use the native encoder from the TTS model (matched codebooks with decoder)
        if let encoder = model.codecEncoder {
            referenceEncoder.setEncoder(encoder)
        }
    }

    /// Whether the codec encoder is ready for voice cloning.
    public var isEncoderLoaded: Bool {
        referenceEncoder.isLoaded
    }

    // MARK: - Prompt Creation

    /// Create a reusable voice clone prompt from reference audio.
    ///
    /// This performs one-time processing of the reference audio:
    /// 1. Extract speaker embedding via ECAPA-TDNN (x-vector)
    /// 2. Encode reference audio via native codec encoder (16 codebooks at 12.5 Hz)
    /// 3. Tokenize reference text via BPE tokenizer (content only)
    /// 4. Tokenize "ref_text" role name for ChatML wrapping during ICL prefill
    ///
    /// The resulting prompt can be reused for unlimited synthesis calls.
    ///
    /// **Note on silence padding:** The `ReferenceEncoder` internally appends 0.5s
    /// of silence to the reference audio before encoding. Do NOT add silence manually
    /// before calling this method — that would result in double silence padding
    /// (1.0s total), wasting prefill positions.
    ///
    /// - Parameters:
    ///   - referenceAudio: Mono audio samples normalized to [-1, 1]. Do NOT pre-pad with silence.
    ///   - referenceText: Transcript of the reference audio.
    ///   - sampleRate: Sample rate of the reference audio (default 24kHz).
    ///   - language: Language identifier (default "english").
    /// - Returns: A reusable `VoiceClonePrompt`.
    public func createPrompt(
        referenceAudio: [Float],
        referenceText: String,
        sampleRate: Int = 24000,
        language: String = "english"
    ) throws -> VoiceClonePrompt {
        guard !referenceAudio.isEmpty else {
            throw VoiceCloningError.emptyReferenceAudio
        }
        guard !referenceText.isEmpty else {
            throw VoiceCloningError.emptyReferenceText
        }
        guard referenceEncoder.isLoaded else {
            throw VoiceCloningError.encoderWeightsNotLoaded
        }

        // 1. Extract speaker embedding: audio → mel → ECAPA-TDNN → [1024]
        let speakerEmbedding = extractSpeakerEmbedding(
            audio: referenceAudio, sampleRate: sampleRate)

        // 2. Encode reference audio: audio → native encoder → [16, T_ref]
        //    (ReferenceEncoder internally appends 0.5s silence before encoding)
        let refCodes = try referenceEncoder.encodeReference(
            audio: referenceAudio, sampleRate: sampleRate)

        // 3. Tokenize reference text (content only — ChatML wrapping applied during prefill)
        let refTokenIds = tokenizer.encode(referenceText)

        // 4. Tokenize the "ref_text" role name for ChatML wrapping
        //    This is the BPE encoding of the string "ref_text" which becomes the role
        //    name in the ChatML conversation template:
        //    <|im_start|>ref_text\n{transcript}<|im_end|>\n
        let refTextRoleTokenIds = tokenizer.encode("ref_text")

        print("[VoiceCloningEngine] refTokenIds: \(refTokenIds.count) tokens")
        print("[VoiceCloningEngine] refTextRoleTokenIds: \(refTextRoleTokenIds) (for ChatML role 'ref_text')")
        print("[VoiceCloningEngine] refCodes shape: \(refCodes.shape)")

        // 5. Compute reference duration (from original audio, before silence padding)
        let referenceDuration = Float(referenceAudio.count) / Float(sampleRate)

        return VoiceClonePrompt(
            speakerEmbedding: speakerEmbedding,
            refCodes: refCodes,
            refTokenIds: refTokenIds,
            refTextRoleTokenIds: refTextRoleTokenIds,
            language: language,
            referenceDuration: referenceDuration
        )
    }

    // MARK: - Synthesis

    /// Synthesize speech with voice cloning.
    ///
    /// - Parameters:
    ///   - text: Target text to synthesize.
    ///   - prompt: Pre-computed voice clone prompt.
    ///   - mode: Cloning mode (`.icl` for highest quality, `.xVectorOnly` for speed).
    ///   - sampling: Sampling configuration.
    /// - Returns: Audio samples at 24kHz.
    public func synthesize(
        text: String,
        prompt: VoiceClonePrompt,
        mode: VoiceCloningMode = .icl,
        sampling: SamplingConfig = .default
    ) throws -> [Float] {
        guard !text.isEmpty else {
            throw VoiceCloningError.emptyTargetText
        }

        // 1. Prepare target text tokens with ChatML template
        let targetTextTokens = prepareTextTokens(text: text)

        // 2. Build prefill embeddings
        //    ICL mode: non-streaming (all text in prefill) with ChatML ref_text role
        //    X-vector mode: streaming (first text + trailing)
        let hiddenSize = config.talker.hiddenSize
        let (prefillEmbeds, trailingTextHidden, ttsPadEmbed) = ICLPrefillBuilder.buildPrefillEmbeddings(
            targetTextTokens: targetTextTokens,
            prompt: prompt,
            mode: mode,
            talker: talker,
            codePredictor: codePredictor,
            hiddenSize: hiddenSize)

        // Materialize lazy computation graph before generation
        eval(prefillEmbeds, trailingTextHidden, ttsPadEmbed)

        // 3. Apply mode-specific sampling adjustments
        let effectiveSampling: SamplingConfig
        switch mode {
        case .icl:
            // Python ICL defaults: temperature=0.9, repetition_penalty=1.05
            // Non-streaming mode means the model has full text context, so lower
            // repetition penalty is sufficient (no risk of text starvation)
            var iclSampling = SamplingConfig(
                temperature: 0.9,
                topK: sampling.topK,
                topP: sampling.topP,
                repetitionPenalty: 1.05,
                maxTokens: sampling.maxTokens
            )
            iclSampling.minP = sampling.minP
            iclSampling.eosLogitBias = sampling.eosLogitBias
            effectiveSampling = iclSampling
        case .xVectorOnly:
            effectiveSampling = sampling
        }

        // 4. Run generation loop
        let result = GenerationLoop.generate(
            prefillEmbeds: prefillEmbeds,
            trailingTextHidden: trailingTextHidden,
            ttsPadEmbed: ttsPadEmbed,
            talker: talker,
            codePredictor: codePredictor,
            config: config,
            sampling: effectiveSampling)

        guard result.numFrames > 0 else {
            throw VoiceCloningError.synthesisFailedEOS
        }

        // 5. Vocoder decode with reference prepending (ICL mode)
        let audio = VocoderPostProcessor.decode(
            generatedCodes: result.allCodebooks,
            prompt: prompt,
            mode: mode,
            decoder: codecDecoder)

        return audio
    }

    // MARK: - Private Helpers

    /// Extract speaker embedding from reference audio.
    private func extractSpeakerEmbedding(audio: [Float], sampleRate: Int) -> MLXArray {
        let mel = SpeakerMel.compute(audio: audio, sampleRate: sampleRate)
        let embedding = speakerEncoder(mel)
        eval(embedding)
        return embedding.squeezed(axis: 0)
    }

    /// Prepare target text tokens with ChatML template.
    ///
    /// Produces: `[<|im_start|>, assistant, \n, ...text..., <|im_end|>, \n, <|im_start|>, assistant, \n]`
    ///
    /// The ICLPrefillBuilder extracts:
    /// - Positions 0-2: role header (used as roleEmbed)
    /// - Positions 3 to len-6: target text content
    /// - Last 5: suffix (excluded from content)
    private func prepareTextTokens(text: String) -> [Int] {
        let imStartId = 151644
        let imEndId = 151645
        let newlineId = 198
        let assistantId = 77091

        var tokens: [Int] = []
        tokens.append(contentsOf: [imStartId, assistantId, newlineId])
        let textTokens = tokenizer.encode(text)
        tokens.append(contentsOf: textTokens)
        tokens.append(contentsOf: [imEndId, newlineId, imStartId, assistantId, newlineId])
        return tokens
    }
}
