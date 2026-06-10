import Foundation
import MLX
import Qwen3TTS

/// Post-processes generated codec tokens for vocoder decoding in ICL mode.
///
/// In ICL mode, reference codec tokens (all 16 codebooks) must be prepended
/// to the generated tokens before vocoder decode. The vocoder's causal decoder
/// then reconstructs audio in the cloned voice. The reference portion of the
/// output waveform is trimmed to yield only the target speech.
///
/// ## Trim Algorithm
///
/// Uses **proportional trimming**: `cut = refFrames / totalFrames * totalSamples`.
///
/// The causal ConvNet decoder has a ~2880-sample startup overhead from the causal
/// convolution warmup chain. This means N codec frames decode to slightly fewer
/// than N × 1920 samples. Proportional trimming automatically accounts for this
/// overhead, whereas fixed `refFrames × 1920` over-trims by ~100ms and clips
/// the start of generated speech.
///
/// After trimming, a 50ms linear fade-in is applied to suppress the decoder's
/// transition transient at the ref-to-gen codec boundary.
enum VocoderPostProcessor {

    /// Decode generated codec tokens to audio, applying ICL reference prepending if needed.
    static func decode(
        generatedCodes: MLXArray,
        prompt: VoiceClonePrompt,
        mode: VoiceCloningMode,
        decoder: SpeechTokenizerDecoder
    ) -> [Float] {
        switch mode {
        case .icl:
            return decodeWithPrepend(
                generatedCodes: generatedCodes,
                prompt: prompt,
                decoder: decoder
            )
        case .xVectorOnly:
            return decoder.decode(codes: generatedCodes)
        }
    }

    /// ICL decode: prepend reference codes, decode, trim reference portion, fade in.
    private static func decodeWithPrepend(
        generatedCodes: MLXArray,
        prompt: VoiceClonePrompt,
        decoder: SpeechTokenizerDecoder
    ) -> [Float] {
        let refFrames = prompt.refCodecFrames
        let genFrames = generatedCodes.dim(2)
        print("[VocoderPostProcessor] refCodes.shape=\(prompt.refCodes.shape), refFrames=\(refFrames), genFrames=\(genFrames)")

        guard refFrames > 0 else {
            return decoder.decode(codes: generatedCodes)
        }

        // Normalize to [16, T_ref] regardless of stored orientation, then batch to [1, 16, T_ref]
        let refCodesBatched = prompt.normalizedRefCodes.expandedDimensions(axis: 0)

        // Prepend reference codes: [1, 16, T_ref] + [1, 16, T_gen] → [1, 16, T_ref + T_gen]
        let combined = concatenated([refCodesBatched, generatedCodes], axis: 2)
        let totalFrames = refFrames + genFrames

        // Decode with chunked processing for memory efficiency
        let waveform = decoder.chunkedDecode(codes: combined, chunkSize: 500)
        let flat = waveform.squeezed()
        eval(flat)
        let fullAudio = flat.asArray(Float.self)

        // Proportional trim: accounts for causal conv startup overhead (~2880 samples).
        //
        // The causal ConvNet decoder produces slightly fewer than N × 1920 samples for
        // N frames due to transpose-conv warmup. Fixed `refFrames × 1920` over-trims by
        // ~107ms. Proportional trimming uses the actual decoded length to compute the
        // correct cut point.
        //
        // Python qwen_tts uses: cut = int(ref_len / total_len * wav_length)
        let cut = Int(Double(refFrames) / Double(totalFrames) * Double(fullAudio.count))

        print("[VocoderPostProcessor] totalAudio=\(fullAudio.count), totalFrames=\(totalFrames)")
        print("[VocoderPostProcessor] proportional cut=\(cut) (refFrames=\(refFrames)/\(totalFrames) × \(fullAudio.count))")

        guard cut >= 0 && cut < fullAudio.count else {
            if cut == 0 { return fullAudio }
            print("[VocoderPostProcessor] WARNING: cut >= audio length, returning empty")
            return []
        }

        var output = Array(fullAudio[cut...])

        // 50ms linear fade-in to suppress decoder transition transient.
        //
        // The vocoder's causal transpose-conv chain produces a brief tonal artifact
        // (~235 Hz, ~200ms) at the ref-to-gen codec boundary. The 0.5s silence
        // padding in the reference mitigates most of this, but a gentle fade-in
        // eliminates the remaining transient without audible impact on speech onset.  NOTE: Changing this to .3 after testing
        let fadeInSamples = min(Int(0.3 * 24000), output.count)  // 1200 samples = 50ms
        for i in 0..<fadeInSamples {
            output[i] *= Float(i) / Float(fadeInSamples)
        }

        return output
    }
}
