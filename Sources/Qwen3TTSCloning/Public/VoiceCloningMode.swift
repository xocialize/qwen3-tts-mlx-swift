/// Voice cloning mode selection.
///
/// Controls the trade-off between cloning quality and inference speed.
public enum VoiceCloningMode: Sendable {
    /// X-vector only mode: injects only the speaker embedding.
    ///
    /// - Prefill: ~10 tokens
    /// - Speaker similarity: ~0.75
    /// - No reference transcript required
    /// - Fastest inference
    case xVectorOnly

    /// Full ICL (In-Context Learning) mode: prepends reference text tokens
    /// and reference codec tokens to the generation prefill.
    ///
    /// - Prefill: ~80+ tokens (depends on reference duration)
    /// - Speaker similarity: ~0.89
    /// - Requires reference transcript
    /// - Higher quality but slower prefill
    case icl
}
