import XCTest
import MLX
@testable import Qwen3TTS
import AudioCommon

/// E1 stochastic sweep: N x-vector clone runs, fresh RNG each, bounded at 250 tokens.
/// Counts how many trajectories fail to hit EOS within 200 frames (~16 s for a ~4 s target).
final class CloneRunawaySweep: XCTestCase {
    func testTenCloneRuns() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QTTS_E1_SWEEP"] == "1", "QTTS_E1_SWEEP=1")

        let model = try await Qwen3TTSModel.fromPretrained(
            modelId: CloneRunawayRepro.model8bit,
            tokenizerModelId: CloneRunawayRepro.tokenizerDir)
        let (refSamples, refRate) = try AudioFileLoader.loadWAV(url: CloneRunawayRepro.refWAV)

        var sampling = SamplingConfig.default
        sampling.maxTokens = 250

        var frames: [Int] = []
        for i in 0..<10 {
            let audio = model.synthesizeWithVoiceClone(
                text: CloneRunawayRepro.text, referenceAudio: refSamples,
                referenceSampleRate: refRate, sampling: sampling)
            let f = audio.count / 1920
            frames.append(f)
            print("[E1-SWEEP] run \(i + 1): \(f) frames (~\(String(format: "%.1f", Double(f)/12.5))s)")
        }
        let runaways = frames.filter { $0 >= 200 }.count
        print("[E1-SWEEP] verdict: \(runaways)/10 runaway (≥200 frames); "
            + "frame counts: \(frames)")
    }
}
