import Foundation
@preconcurrency import MLX

/// A pre-computed, reusable voice clone prompt.
///
/// Contains all the extracted information from a reference audio clip needed
/// for voice cloning: speaker embedding (x-vector), codec token indices,
/// reference text BPE tokens, and ChatML role tokens. Once created, can be
/// reused for unlimited synthesis calls without re-encoding the reference audio.
///
/// Serialize to disk via ``PromptSerializer`` for persistent storage.
///
/// ## ChatML Role Wrapping (ICL mode)
///
/// The `refTextRoleTokenIds` field contains the BPE token IDs for the string
/// "ref_text" — the ChatML role name used to wrap the reference transcript.
/// This is computed once during `createPrompt()` and stored for reuse.
///
/// During ICL prefill construction, the reference text is wrapped as:
/// ```
/// <|im_start|>{refTextRoleTokenIds}\n{refTokenIds}<|im_end|>\n
/// ```
///
/// This role boundary tells the model "this text is reference context that has
/// already been spoken" — preventing it from regenerating the reference speech
/// before the target text.
public struct VoiceClonePrompt: @unchecked Sendable {
    /// 1024-dim speaker embedding from ECAPA-TDNN. Shape: [1024]
    public let speakerEmbedding: MLXArray

    /// 16-codebook RVQ token indices from the native codec encoder. Shape: [16, T_ref]
    public let refCodes: MLXArray

    /// BPE token IDs for the reference text transcript (content only, no ChatML wrapping).
    public let refTokenIds: [Int]

    /// BPE token IDs for the string "ref_text" (the ChatML role name).
    /// Computed once during createPrompt() and stored for reuse.
    /// Used by ICLPrefillBuilder to construct the ChatML-wrapped reference text.
    public let refTextRoleTokenIds: [Int]

    /// Language identifier (e.g., "english", "chinese").
    public let language: String

    /// Duration of the reference audio in seconds, used for vocoder trimming.
    public let referenceDuration: Float

    public init(
        speakerEmbedding: MLXArray,
        refCodes: MLXArray,
        refTokenIds: [Int],
        refTextRoleTokenIds: [Int] = [],
        language: String,
        referenceDuration: Float
    ) {
        self.speakerEmbedding = speakerEmbedding
        self.refCodes = refCodes
        self.refTokenIds = refTokenIds
        self.refTextRoleTokenIds = refTextRoleTokenIds
        self.language = language
        self.referenceDuration = referenceDuration
    }

    /// Number of reference codec frames (temporal dimension).
    ///
    /// Handles both standard `[16, T_ref]` and potentially transposed `[T_ref, 16]` shapes.
    /// The 16-codebook dimension is always exactly 16, so we use that to identify orientation.
    public var refCodecFrames: Int {
        if refCodes.ndim == 2 {
            let d0 = refCodes.dim(0)
            let d1 = refCodes.dim(1)
            if d0 == 16 { return d1 }      // [16, T_ref] — standard
            else if d1 == 16 { return d0 }  // [T_ref, 16] — transposed
        }
        return refCodes.dim(refCodes.ndim - 1)
    }

    /// Returns refCodes normalized to `[16, T_ref]` shape regardless of stored orientation.
    public var normalizedRefCodes: MLXArray {
        if refCodes.ndim == 2, refCodes.dim(0) != 16, refCodes.dim(1) == 16 {
            return refCodes.transposed()
        }
        return refCodes
    }
}
