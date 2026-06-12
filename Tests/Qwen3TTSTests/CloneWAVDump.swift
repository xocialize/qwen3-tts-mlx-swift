import XCTest
import MLX
@testable import Qwen3TTS
import AudioCommon

/// E8 verification: write post-fix clone + zero-shot WAVs for listening, with head/tail stats.
final class CloneWAVDump: XCTestCase {
    func testDumpWAVs() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QTTS_E8_WAV"] == "1", "QTTS_E8_WAV=1")

        let model = try await Qwen3TTSModel.fromPretrained(
            modelId: CloneRunawayRepro.model8bit,
            tokenizerModelId: CloneRunawayRepro.tokenizerDir)
        let (refSamples, refRate) = try AudioFileLoader.loadWAV(url: CloneRunawayRepro.refWAV)

        var sampling = SamplingConfig.default
        sampling.maxTokens = 250

        let zs = model.synthesize(text: CloneRunawayRepro.text, sampling: sampling)
        try WAVWriter.write(samples: zs, sampleRate: 24000,
                            to: URL(fileURLWithPath: NSHomeDirectory() + "/Desktop/E8-fixed-zeroshot.wav"))

        let cl = model.synthesizeWithVoiceClone(
            text: CloneRunawayRepro.text, referenceAudio: refSamples,
            referenceSampleRate: refRate, sampling: sampling)
        try WAVWriter.write(samples: cl, sampleRate: 24000,
                            to: URL(fileURLWithPath: NSHomeDirectory() + "/Desktop/E8-fixed-clone-samantha.wav"))

        // Head profile: RMS of the first 6 × 100 ms windows (lead-in check).
        for (label, x) in [("zero-shot", zs), ("clone", cl)] {
            let w = 2400
            let heads = (0..<6).map { i -> Float in
                let seg = Array(x[min(i * w, x.count)..<min((i + 1) * w, x.count)])
                return (seg.reduce(0) { $0 + $1 * $1 } / Float(max(seg.count, 1))).squareRoot()
            }
            print("[E8-WAV] \(label) head 100ms-RMS:", heads.map { String(format: "%.3f", $0) },
                  "total \(String(format: "%.2f", Double(x.count) / 24000))s")
        }
    }
}
