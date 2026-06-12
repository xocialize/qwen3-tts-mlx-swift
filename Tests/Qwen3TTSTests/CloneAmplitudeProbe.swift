import XCTest
import MLX
@testable import Qwen3TTS
import AudioCommon

/// E8 probe: raw float amplitude of the decoded waveform on the clone vs zero-shot paths.
/// If clone floats exceed ±1.0, the int16 WAV write hard-clips → the "tonal buzz".
final class CloneAmplitudeProbe: XCTestCase {
    func testDecodedAmplitude() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QTTS_E8_PROBE"] == "1", "QTTS_E8_PROBE=1")

        let model = try await Qwen3TTSModel.fromPretrained(
            modelId: CloneRunawayRepro.model8bit,
            tokenizerModelId: CloneRunawayRepro.tokenizerDir)
        let (refSamples, refRate) = try AudioFileLoader.loadWAV(url: CloneRunawayRepro.refWAV)

        var sampling = SamplingConfig.default
        sampling.maxTokens = 250

        func stats(_ x: [Float], _ label: String) {
            let peak = x.map { abs($0) }.max() ?? 0
            let rms = (x.reduce(Float(0)) { $0 + $1 * $1 } / Float(max(x.count, 1))).squareRoot()
            let over = x.filter { abs($0) > 1.0 }.count
            print("[E8-PROBE] \(label): \(x.count) samples · peak \(peak) · rms \(rms) "
                + "· samples>|1.0|: \(over) (\(String(format: "%.1f", 100 * Double(over) / Double(max(x.count, 1))))%)")
        }

        let zs = model.synthesize(text: CloneRunawayRepro.text, sampling: sampling)
        stats(zs, "zero-shot")

        let cl = model.synthesizeWithVoiceClone(
            text: CloneRunawayRepro.text, referenceAudio: refSamples,
            referenceSampleRate: refRate, sampling: sampling)
        stats(cl, "x-vector ")
    }
}
