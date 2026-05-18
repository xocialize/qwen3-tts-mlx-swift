import Foundation
import MLX

/// Sampling configuration for TTS generation
public struct SamplingConfig: Sendable {
    public var temperature: Float = 0.5
    public var topK: Int = 50
    public var topP: Float = 1.0
    public var minP: Float = 0.0
    public var repetitionPenalty: Float = 1.05
    public var maxTokens: Int = 4096
    /// Additive bias to EOS logit (applied after temperature, before sampling).
    /// Positive values make the model more likely to stop. Useful for CustomVoice
    /// models that under-score EOS for certain languages.
    public var eosLogitBias: Float = 0.0
    /// Optional random seed for reproducible sampling. When set, the MLX random state
    /// is seeded before each generation for consistent voice timbre across segments.
    /// Recommended for VoiceDesign to prevent voice drift between chunks.
    /// Use a per-speaker seed (e.g., hash of speakerID) for character consistency.
    public var seed: UInt64? = nil

    public init() {}

    public init(
        temperature: Float = 0.5,
        topK: Int = 50,
        topP: Float = 1.0,
        repetitionPenalty: Float = 1.05,
        maxTokens: Int = 4096,
        eosLogitBias: Float = 0.0,
        seed: UInt64? = nil
    ) {
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.maxTokens = maxTokens
        self.eosLogitBias = eosLogitBias
        self.seed = seed
    }

    public static var `default`: SamplingConfig { SamplingConfig() }
    public static var greedy: SamplingConfig { SamplingConfig(temperature: 0, topK: 1) }

    /// VoiceDesign preset: optimized for creative voice generation from text descriptions.
    /// Higher temperature (0.8) for more expressive/varied voices, top-p nucleus sampling (0.9).
    /// EOS bias set to 0.0 to match HuggingFace demo behavior — the demo produces clean output
    /// without any bias. Tonal tail trimming in Qwen3TTS.swift handles runaway cases.
    /// Reference: Community testing shows temperature <0.7 sounds robotic for VoiceDesign.
    /// Note: Set `seed` per-speaker for consistent character voices across segments.
    public static var voiceDesign: SamplingConfig {
        SamplingConfig(temperature: 0.8, topK: 50, topP: 0.9, repetitionPenalty: 1.05, maxTokens: 2048, eosLogitBias: 0.0)
    }

    /// Create a config with a deterministic seed derived from a speaker identifier.
    /// Use this for consistent character voices across segments and projects.
    public static func withSpeakerSeed(_ speakerID: String, base: SamplingConfig = .default) -> SamplingConfig {
        var config = base
        config.seed = speakerSeed(from: speakerID)
        return config
    }

    /// Derive a stable seed from a speaker identifier string.
    /// Uses FNV-1a hash for good distribution and consistency across runs.
    ///
    /// **Important:** The seed is truncated to 32-bit to avoid MLX seeding issues.
    /// Testing revealed that certain large 64-bit seeds (e.g., >5×10^18) cause
    /// deterministic Gumbel noise patterns that systematically avoid the EOS token,
    /// resulting in runaway generation. Truncating to 32-bit keeps seeds in a
    /// "safe" range where EOS is properly sampled.
    public static func speakerSeed(from speakerID: String) -> UInt64 {
        // FNV-1a 64-bit hash
        var hash: UInt64 = 14695981039346656037  // FNV offset basis
        for byte in speakerID.utf8 {
            hash ^= UInt64(byte)
            hash &*= 1099511628211  // FNV prime
        }
        // Truncate to 32-bit to avoid MLX seeding issues with large seeds.
        // Large 64-bit seeds can create Gumbel noise patterns that never favor EOS.
        return hash & 0xFFFFFFFF
    }
}

/// Seed the MLX random state once at the START of generation.
/// Call this ONCE before the generation loop begins, not per-token.
///
/// Per-token seeding creates identical Gumbel noise on every step, which can
/// cause certain seeds to systematically avoid EOS tokens (runaway generation).
/// Seeding once at the start allows the Gumbel sequence to vary naturally.
///
/// - Parameter seed: The seed value. Use `SamplingConfig.speakerSeed(from:)` to
///   derive a stable seed from a speaker ID for consistent voice timbre.
public func seedSamplingRNG(_ seed: UInt64) {
    MLXRandom.seed(seed)
}

/// Sample a token from logits using temperature, top-k, top-p, and repetition penalty.
/// EOS protection: the EOS logit is saved before top-k/top-p filtering and restored after.
/// Uses top-k filtered multinomial sampling with explicit probability masking to avoid
/// MLX categorical sampling bugs with zero-probability tokens.
///
/// **Important:** If you want reproducible sampling, call `seedSamplingRNG()` once
/// BEFORE the generation loop starts. Do NOT pass a seed to this function per-token.
public func sampleToken(
    logits: MLXArray,
    config: SamplingConfig,
    generatedTokens: [Int32] = [],
    suppressRange: (Int, Int)? = nil,
    eosTokenId: Int? = nil
) -> Int32 {
    // NOTE: Seeding was moved to seedSamplingRNG() called once at generation start.
    // Per-token seeding caused runaway generation for certain seed values.

    // logits: [1, 1, vocab] or [vocab] — work with last dim
    var logits = logits.squeezed().asType(.float32)  // [vocab]
    let vocabSize = logits.dim(0)

    // 1. Token suppression: set range to -inf (except EOS)
    if let (start, end) = suppressRange, start < end, start >= 0, end <= vocabSize {
        let indices = MLXArray(0..<Int32(vocabSize))
        let geStart = indices .>= MLXArray(Int32(start))
        let ltEnd = indices .< MLXArray(Int32(end))
        var suppressMask = logicalAnd(geStart, ltEnd)

        if let eos = eosTokenId, eos >= start, eos < end {
            let notEos = indices .!= MLXArray(Int32(eos))
            suppressMask = logicalAnd(suppressMask, notEos)
        }

        logits = MLX.where(suppressMask, MLXArray(Float(-1e9)), logits)
    }

    // 2. Repetition penalty
    if config.repetitionPenalty != 1.0 && !generatedTokens.isEmpty {
        let uniqueTokens = Array(Set(generatedTokens))
        let indices = MLXArray(0..<Int32(vocabSize))
        var penaltyMask = indices .== Int32(-1)  // all false
        for token in uniqueTokens {
            penaltyMask = logicalOr(penaltyMask, indices .== token)
        }

        let penalty = MLXArray(config.repetitionPenalty)
        let penalizedPos = logits / penalty
        let penalizedNeg = logits * penalty
        let penalized = MLX.where(logits .< MLXArray(Float(0)), penalizedNeg, penalizedPos)
        logits = MLX.where(penaltyMask, penalized, logits)
    }

    // 3. Greedy decoding
    if config.temperature <= 0 {
        return argMax(logits).item(Int32.self)
    }

    // 4. Save EOS logit BEFORE any filtering (matching reference implementation)
    // This preserves the original EOS probability independent of temperature, preventing
    // premature termination at low temp or runaway generation at high temp.
    var savedEosLogit: MLXArray? = nil
    if let eos = eosTokenId, eos >= 0, eos < vocabSize {
        savedEosLogit = logits[eos]
    }

    // 5. Top-k filtering (BEFORE temperature, matching Python reference!)
    // Filtering on original logits gives more meaningful top-k selection,
    // especially important for VoiceDesign where higher temperature is used.
    if config.topK > 0 && config.topK < vocabSize {
        let sorted = MLX.sorted(logits)
        let threshold = sorted[vocabSize - config.topK]
        logits = MLX.where(logits .< threshold, MLXArray(Float(-1e9)), logits)
    }

    // 6. Top-p (nucleus) filtering (also before temperature)
    if config.topP < 1.0 {
        let sortedIndices = argSort(logits)
        let sortedLogits = logits[sortedIndices]
        let probs = softmax(sortedLogits)
        let cumProbs = cumsum(probs)

        let sortedMask = cumProbs - probs .> MLXArray(config.topP)
        let filteredLogits = MLX.where(sortedMask, MLXArray(Float(-1e9)), sortedLogits)

        let unsortIndices = argSort(sortedIndices)
        logits = filteredLogits[unsortIndices]
    }

    // 7. Restore EOS logit (original, unscaled value) + apply EOS bias
    if let eos = eosTokenId, let eosLogit = savedEosLogit, eos >= 0, eos < vocabSize {
        let biasedEos = config.eosLogitBias != 0
            ? eosLogit + MLXArray(config.eosLogitBias)
            : eosLogit
        let indices = MLXArray(0..<Int32(vocabSize))
        let eosMask = indices .== MLXArray(Int32(eos))
        logits = MLX.where(eosMask, biasedEos, logits)
    }

    // 8. Apply temperature
    logits = logits / MLXArray(config.temperature)

    // 9. Multinomial sampling via Gumbel-max trick
    // argmax(logits + Gumbel) ~ Categorical(softmax(logits))
    // This avoids MLX categorical bugs where zero-probability tokens can be sampled.
    let gumbel = MLXRandom.gumbel(logits.shape)
    let perturbedLogits = logits + gumbel
    return argMax(perturbedLogits).item(Int32.self)
}

/// Lazy version of sampleToken for code predictor: returns MLXArray (no GPU sync).
///
/// Identical sampling logic (temperature, top-k, Gumbel-max) but returns the argmax result
/// as a lazy MLXArray scalar instead of calling `.item()`. This allows chaining 15 CP group
/// predictions into one lazy computation graph, evaluated with a single `eval()` at the end.
///
/// Omits repetition penalty, token suppression, and EOS protection (not needed for CP).
/// For reproducible sampling, call `seedSamplingRNG()` once at generation start.
public func sampleTokenLazy(
    logits: MLXArray,
    config: SamplingConfig
) -> MLXArray {
    // NOTE: Seeding was moved to seedSamplingRNG() called once at generation start.

    var logits = logits.squeezed().asType(.float32)  // [vocab]

    if config.temperature <= 0 {
        return argMax(logits).asType(.int32)
    }

    // Top-k filtering BEFORE temperature (matching Python reference)
    let vocabSize = logits.dim(0)
    if config.topK > 0 && config.topK < vocabSize {
        let sorted = MLX.sorted(logits)
        let threshold = sorted[vocabSize - config.topK]
        logits = MLX.where(logits .< threshold, MLXArray(Float(-1e9)), logits)
    }

    // Apply temperature after filtering
    logits = logits / MLXArray(config.temperature)

    let gumbel = MLXRandom.gumbel(logits.shape)
    let perturbedLogits = logits + gumbel
    return argMax(perturbedLogits).asType(.int32)  // scalar MLXArray — NO .item()!
}

/// Batch-sample tokens from logits for B items simultaneously.
/// Supports temperature, top-k, top-p, token suppression, and EOS protection.
/// Finished items receive `padToken` instead of a sampled token.
/// Repetition penalty is skipped in batch mode (requires per-item history tracking).
///
/// For reproducible sampling, call `seedSamplingRNG()` once at generation start.
///
/// - Parameters:
///   - logits: `[B, 1, vocab]` or `[B, vocab]`
///   - config: Sampling configuration (temperature, topK, topP)
///   - finishedMask: `[B]` bool — true = item already finished
///   - padToken: Token to feed back for finished items
///   - suppressRange: Optional range of token IDs to suppress (except EOS)
///   - eosTokenId: EOS token ID (protected from suppression/filtering)
/// - Returns: `[B]` Int32 sampled tokens
public func sampleTokensBatch(
    logits: MLXArray,
    config: SamplingConfig,
    finishedMask: MLXArray,
    padToken: Int32,
    suppressRange: (Int, Int)? = nil,
    eosTokenId: Int? = nil
) -> MLXArray {
    // NOTE: Seeding was moved to seedSamplingRNG() called once at generation start.

    // Squeeze to [B, vocab]
    var logits2d: MLXArray
    if logits.ndim == 3 {
        logits2d = logits.squeezed(axis: 1).asType(.float32)
    } else {
        logits2d = logits.asType(.float32)
    }
    let vocabSize = logits2d.dim(1)

    // 1. Token suppression: broadcast [vocab] mask over [B, vocab]
    if let (start, end) = suppressRange, start < end, start >= 0, end <= vocabSize {
        let indices = MLXArray(0..<Int32(vocabSize))
        let geStart = indices .>= MLXArray(Int32(start))
        let ltEnd = indices .< MLXArray(Int32(end))
        var suppressMask = logicalAnd(geStart, ltEnd)  // [vocab]

        if let eos = eosTokenId, eos >= start, eos < end {
            let notEos = indices .!= MLXArray(Int32(eos))
            suppressMask = logicalAnd(suppressMask, notEos)
        }

        // Broadcast [vocab] over [B, vocab]
        logits2d = MLX.where(suppressMask, MLXArray(Float(-1e9)), logits2d)
    }

    // 2. Greedy decoding
    if config.temperature <= 0 {
        let tokens = argMax(logits2d, axis: 1).asType(.int32)  // [B]
        return MLX.where(finishedMask, MLXArray(padToken), tokens)
    }

    // 3. Save EOS logit per row BEFORE any filtering (matching reference)
    var savedEosCol: MLXArray? = nil
    if let eos = eosTokenId, eos >= 0, eos < vocabSize {
        savedEosCol = logits2d[0..., eos..<(eos + 1)]  // [B, 1]
    }

    // 4. Top-k filtering (BEFORE temperature, matching Python reference!)
    if config.topK > 0 && config.topK < vocabSize {
        let sorted = MLX.sorted(logits2d, axis: 1)  // [B, vocab]
        let threshold = sorted[0..., (vocabSize - config.topK)..<(vocabSize - config.topK + 1)]  // [B, 1]
        logits2d = MLX.where(logits2d .< threshold, MLXArray(Float(-1e9)), logits2d)
    }

    // 5. Top-p filtering (also before temperature)
    if config.topP < 1.0 {
        let sortedIndices = argSort(logits2d, axis: 1)  // [B, vocab]
        let sortedLogits = takeAlong(logits2d, sortedIndices, axis: 1)  // [B, vocab]
        let probs = softmax(sortedLogits, axis: 1)
        let cumProbs = cumsum(probs, axis: 1)

        let sortedMask = cumProbs - probs .> MLXArray(config.topP)
        let filteredSorted = MLX.where(sortedMask, MLXArray(Float(-1e9)), sortedLogits)

        let unsortIndices = argSort(sortedIndices, axis: 1)
        logits2d = takeAlong(filteredSorted, unsortIndices, axis: 1)
    }

    // 6. Restore EOS logit (original, unscaled value) + apply EOS bias
    if let eos = eosTokenId, let eosCol = savedEosCol, eos >= 0, eos < vocabSize {
        let biasedEos = config.eosLogitBias != 0
            ? eosCol + MLXArray(config.eosLogitBias)
            : eosCol
        let indices = MLXArray(0..<Int32(vocabSize))
        let eosMask = indices .== MLXArray(Int32(eos))  // [vocab], broadcasts over [B, vocab]
        logits2d = MLX.where(eosMask, biasedEos, logits2d)
    }

    // 7. Apply temperature (after filtering)
    logits2d = logits2d / MLXArray(config.temperature)

    // 8. Gumbel-max sampling: argmax(logits + Gumbel) ~ Categorical(softmax(logits))
    let gumbel = MLXRandom.gumbel(logits2d.shape)
    let perturbed = logits2d + gumbel
    let sampledTokens = argMax(perturbed, axis: 1).asType(.int32)  // [B]

    // 9. Replace finished items with pad token
    return MLX.where(finishedMask, MLXArray(padToken), sampledTokens)
}
