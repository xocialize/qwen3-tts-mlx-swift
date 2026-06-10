import MLX
import Qwen3TTS

/// Reimplemented autoregressive generation loop for voice cloning.
///
/// This reimplements qwen3-asr-swift's private `generateWithCodePredictor()` and
/// `predictCodebooksForTimestep()` using only public APIs from `TalkerModel`,
/// `CodePredictorModel`, and the sampling functions.
///
/// ## Two-Level Generation
///
/// 1. **Talker** generates first-codebook (codebook-0) tokens autoregressively.
///    Each step input = text_embed + codec_embed (element-wise sum).
///
/// 2. **CodePredictor** generates the remaining 15 codebook tokens for each timestep,
///    autoregressively across groups using its own 5-layer transformer (fresh KV cache
///    per timestep).
enum GenerationLoop {

    /// Result of the generation loop.
    struct GenerationResult {
        /// All 16 codebook tokens. Shape: `[1, 16, T_gen]`
        let allCodebooks: MLXArray
        /// Number of generated frames (T_gen).
        let numFrames: Int
    }

    /// Run the autoregressive generation loop.
    ///
    /// - Parameters:
    ///   - prefillEmbeds: Pre-built prefill embeddings from ICLPrefillBuilder. Shape: `[1, prefillLen, D]`
    ///   - trailingTextHidden: Remaining target text embeddings to feed one-per-step. Shape: `[1, trailLen, D]`
    ///   - ttsPadEmbed: TTS pad embedding for positions after trailing text. Shape: `[1, 1, D]`
    ///   - talker: The Talker model.
    ///   - codePredictor: The CodePredictor model.
    ///   - config: Talker config (for numLayers, numCodeGroups).
    ///   - sampling: Sampling configuration.
    /// - Returns: Generated codebook tokens and frame count.
    static func generate(
        prefillEmbeds: MLXArray,
        trailingTextHidden: MLXArray,
        ttsPadEmbed: MLXArray,
        talker: TalkerModel,
        codePredictor: CodePredictorModel,
        config: Qwen3TTSConfig,
        sampling: SamplingConfig
    ) -> GenerationResult {
        let safeMaxTokens = min(sampling.maxTokens, 500)
        let numCodeGroups = config.codePredictor.numCodeGroups  // 16

        var generatedFirstCodebook: [Int32] = []
        var generatedAllCodebooks: [[Int32]] = (0..<numCodeGroups).map { _ in [] }

        let cpSamplingConfig = SamplingConfig(temperature: sampling.temperature, topK: sampling.topK)
        let prefillLen = prefillEmbeds.dim(1)

        // Dynamic suppress range based on codec vocab size (matches reference implementation)
        let vocabSize = config.talker.codecVocabSize
        let suppressRange = (vocabSize - 1024, vocabSize)

        // --- Prefill ---
        var (logits, hiddenStates, talkerCache) = talker(
            inputsEmbeds: prefillEmbeds,
            offset: MLXArray(Int32(0)),
            cache: nil)

        // Sample first token from last position
        let lastLogits = logits[0..., (prefillLen - 1)..<prefillLen, 0...]
        var nextToken = sampleToken(
            logits: lastLogits,
            config: sampling,
            generatedTokens: generatedFirstCodebook,
            suppressRange: suppressRange,
            eosTokenId: CodecTokens.codecEos)

        if nextToken == Int32(CodecTokens.codecEos) {
            return GenerationResult(allCodebooks: MLXArray.zeros([1, 16, 0]), numFrames: 0)
        }

        generatedFirstCodebook.append(nextToken)
        generatedAllCodebooks[0].append(nextToken)

        // Get hidden state for code predictor
        let lastHidden = hiddenStates[0..., (prefillLen - 1)..<prefillLen, 0...]  // [1, 1, D]

        // Run code predictor for remaining 15 codebook tokens
        var codeTokens = predictCodebooks(
            hiddenState: lastHidden,
            firstCodebookToken: nextToken,
            codePredictor: codePredictor,
            talker: talker,
            numCodeGroups: numCodeGroups,
            cpSamplingConfig: cpSamplingConfig)
        for (i, token) in codeTokens.enumerated() {
            generatedAllCodebooks[i + 1].append(token)
        }

        var trailingIdx = 0
        var step = prefillLen

        // --- Autoregressive generation ---
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

            // Codec side: sum codebook-0 embed + all 15 group embeddings
            let codecEmbed = talker.embedCodec(
                MLXArray([nextToken]).expandedDimensions(axis: 0))  // [1, 1, D]
                + codePredictor.batchEmbedAllGroups(codeTokens)     // [1, 1, D]

            // Next input = text + codec (element-wise sum)
            let stepEmbeds = textEmbed + codecEmbed  // [1, 1, D]

            let newResult = talker(
                inputsEmbeds: stepEmbeds,
                offset: MLXArray(Int32(step)),
                cache: talkerCache)
            let stepLogits = newResult.0
            hiddenStates = newResult.1
            talkerCache = newResult.2

            nextToken = sampleToken(
                logits: stepLogits,
                config: sampling,
                generatedTokens: generatedFirstCodebook,
                suppressRange: suppressRange,
                eosTokenId: CodecTokens.codecEos)

            if nextToken == Int32(CodecTokens.codecEos) { break }

            generatedFirstCodebook.append(nextToken)
            generatedAllCodebooks[0].append(nextToken)

            // Code predictor for this timestep
            codeTokens = predictCodebooks(
                hiddenState: hiddenStates,  // [1, 1, D]
                firstCodebookToken: nextToken,
                codePredictor: codePredictor,
                talker: talker,
                numCodeGroups: numCodeGroups,
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
        let estAudioSec = Double(numFrames) / 12.5
        print("  Talker done: \(numFrames) codec tokens (~\(String(format: "%.1f", estAudioSec))s audio)")

        // Stack all codebooks: [1, 16, T]
        let codebookArrays = generatedAllCodebooks.map { tokens in
            MLXArray(tokens).expandedDimensions(axis: 0)  // [1, T]
        }
        let allCodebooks = stacked(codebookArrays, axis: 1)  // [1, 16, T]

        return GenerationResult(allCodebooks: allCodebooks, numFrames: numFrames)
    }

    /// Predict codebook tokens 1-15 for a single timestep using the CodePredictor.
    ///
    /// Uses the public `codePredictor(inputsEmbeds:, groupIndex:, cache:)` API
    /// which runs the full forward pass (layers + norm + lmHead) for each group.
    ///
    /// - Parameters:
    ///   - hiddenState: Talker hidden state for this timestep. Shape: `[1, 1, D]`
    ///   - firstCodebookToken: The codebook-0 token sampled by the Talker.
    ///   - codePredictor: The CodePredictor model.
    ///   - talker: The Talker model (for codebook-0 embedding).
    ///   - numCodeGroups: Total number of codebook groups (16).
    ///   - cpSamplingConfig: Sampling config for code predictor.
    /// - Returns: Array of 15 tokens for codebook groups 1-15.
    private static func predictCodebooks(
        hiddenState: MLXArray,
        firstCodebookToken: Int32,
        codePredictor: CodePredictorModel,
        talker: TalkerModel,
        numCodeGroups: Int,
        cpSamplingConfig: SamplingConfig
    ) -> [Int32] {
        // First codebook embedding (from talker's codec embedding table)
        let code0Embed = talker.embedCodec(
            MLXArray([firstCodebookToken]).expandedDimensions(axis: 0))  // [1, 1, D]

        // Prefill: [hidden_state, code_0_embed] — length 2
        let prefillInput = concatenated([hiddenState, code0Embed], axis: 1)  // [1, 2, D]

        // Predict group 0 (= codebook 1 overall, since codebook 0 is from the Talker)
        let (cpLogits0, cpCache0) = codePredictor(
            inputsEmbeds: prefillInput, groupIndex: 0, cache: nil)

        // Sample from last position
        let lastCpLogits = cpLogits0[0..., 1..<2, 0...]
        var prevTokenArray = sampleTokenLazy(logits: lastCpLogits, config: cpSamplingConfig)
        var lazyTokens: [MLXArray] = [prevTokenArray]
        var cpCache = cpCache0

        // Remaining 14 codebook groups — chained as lazy MLXArray computations
        for groupIdx in 1..<(numCodeGroups - 1) {
            // Embed previous group's token
            let prevEmbed = codePredictor.embedCodecGroup(
                prevTokenArray.reshaped(1, 1),
                groupIndex: groupIdx - 1)  // [1, 1, D]

            // Run through code predictor for next group
            let (groupLogits, newCache) = codePredictor(
                inputsEmbeds: prevEmbed, groupIndex: groupIdx, cache: cpCache)
            cpCache = newCache

            prevTokenArray = sampleTokenLazy(logits: groupLogits, config: cpSamplingConfig)
            lazyTokens.append(prevTokenArray)
        }

        // ONE eval to materialize the entire 15-group computation graph
        let tokenStack = stacked(lazyTokens)  // [15]
        eval(tokenStack)
        return tokenStack.asArray(Int32.self)
    }
}
