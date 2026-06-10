import Foundation
import MLX
import Qwen3TTS

/// Wraps ``SpeechTokenizerEncoder`` for reference audio encoding.
///
/// Uses the native encoder from the TTS package, which shares matched codebooks
/// with ``SpeechTokenizerDecoder``. This eliminates the encoder/decoder codebook
/// mismatch that caused clipping artifacts with the separate MimiCodecEncoder.
///
/// Appends 0.5 seconds of silence to reference audio before encoding to prevent
/// phoneme bleed at the reference-to-generated boundary (matches faster-qwen3-tts
/// default behavior).
public final class ReferenceEncoder: @unchecked Sendable {
    private var encoder: SpeechTokenizerEncoder?

    /// Whether encoder is available for use.
    public var isLoaded: Bool { encoder != nil }

    public init() {}

    /// Set the native encoder (loaded with the TTS model).
    public func setEncoder(_ encoder: SpeechTokenizerEncoder) {
        self.encoder = encoder
    }

    /// Encode reference audio into codec tokens.
    ///
    /// - Parameters:
    ///   - audio: Mono audio samples, normalized to [-1, 1].
    ///   - sampleRate: Sample rate of the input audio.
    /// - Returns: Codec indices of shape `[16, T_ref]`.
    public func encodeReference(audio: [Float], sampleRate: Int = 24000) throws -> MLXArray {
        guard let encoder = encoder else {
            throw VoiceCloningError.encoderWeightsNotLoaded
        }
        guard !audio.isEmpty else {
            throw VoiceCloningError.emptyReferenceAudio
        }

        // Resample to 24kHz if needed
        var samples = audio
        if sampleRate != 24000 {
            samples = resample(samples, from: sampleRate, to: 24000)
        }

        // Safety cap: truncate to 30s max to prevent oversized prefill sequences.
        //
        // CRITICAL: The previous 10s cap silently truncated reference audio while
        // the transcript still covered the full duration. This caused ref code / ref
        // text misalignment in the ICL prefill — the model saw transcript for audio
        // it never encoded, producing garbled output (ChatML tokens vocalized,
        // reference text regenerated before target speech).
        //
        // 30s accommodates the recommended 10-15s reference range plus generous
        // margin. References longer than 30s risk oversized prefill sequences
        // (30s × 12.5 Hz = 375 codec frames ≈ 375 prefill positions).
        let maxSamples = 30 * 24000
        if samples.count > maxSamples {
            let capDuration = Double(maxSamples) / 24000.0
            let fullDuration = Double(samples.count) / 24000.0
            print("[ReferenceEncoder] WARNING: Truncating reference from \(String(format: "%.1f", fullDuration))s to \(String(format: "%.1f", capDuration))s")
            print("  The reference transcript should be shortened to match.")
            samples = Array(samples.prefix(maxSamples))
        }

        // Append 0.5s silence to prevent phoneme bleed at the reference-to-generated
        // boundary. The model's final prefill token is conditioned on the reference's
        // last phoneme — silence ensures clean acoustic context for generation start.
        // Matches faster-qwen3-tts default behavior (append_silence=True).
        // Adds ~6 codec frames (0.5 × 12.5 = 6.25) to ref_codes, automatically
        // included in the token-level trim.
        let silenceSamples = Int(0.5 * 24000)  // 12,000 samples
        samples.append(contentsOf: [Float](repeating: 0, count: silenceSamples))

        let audioArray = MLXArray(samples)
        let codes = encoder.encode(audio: audioArray)
        MLX.eval(codes)
        return codes  // [16, T_ref]
    }

    /// Simple linear resampling (nearest-neighbor interpolation).
    private func resample(_ audio: [Float], from sourceSR: Int, to targetSR: Int) -> [Float] {
        let ratio = Double(targetSR) / Double(sourceSR)
        let outputLen = Int(Double(audio.count) * ratio)
        var output = [Float](repeating: 0, count: outputLen)
        for i in 0..<outputLen {
            let srcIdx = min(Int(Double(i) / ratio), audio.count - 1)
            output[i] = audio[srcIdx]
        }
        return output
    }
}
