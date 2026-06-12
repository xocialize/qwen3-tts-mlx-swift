import XCTest
import MLX
@testable import Qwen3TTS
import AudioCommon

/// E1 repro (EXTERNAL-RESOLVE): x-vector cloning runaway — Talker never emits EOS with
/// a reference voice while zero-shot stops fine on the same checkpoint.
/// Gated: QTTS_E1_REPRO=1 (needs the Cmlx metallib bundle in .build/debug/ and the
/// 8-bit checkpoint at DEV_ARCHIVE). Caps generation at 250 tokens so the repro is
/// bounded: the target sentence is ~4 s ≈ 50 frames — anything > 200 without EOS = runaway.
final class CloneRunawayRepro: XCTestCase {
    static let model8bit = "/Volumes/DEV_ARCHIVE/models/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
    static let tokenizerDir = "/Volumes/DEV_VOL1/anime-studio/models/Qwen3-TTS-Tokenizer-12Hz"
    static let refWAV = URL(fileURLWithPath: "/tmp/ref_voice_samantha.wav")
    static let text = "The morning light spilled across the quiet harbor as the boats began to stir."

    func testZeroShotStopsButCloneRunsAway() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QTTS_E1_REPRO"] == "1", "QTTS_E1_REPRO=1")

        let model = try await Qwen3TTSModel.fromPretrained(
            modelId: Self.model8bit, tokenizerModelId: Self.tokenizerDir)

        var sampling = SamplingConfig.default
        sampling.maxTokens = 250   // bounded repro

        // Control: zero-shot on the same checkpoint — expect EOS well under the cap.
        let t0 = Date()
        let zeroShot = model.synthesize(text: Self.text, sampling: sampling)
        let zsFrames = zeroShot.count / 1920
        print("[E1-REPRO] zero-shot: \(zsFrames) frames (~\(Double(zsFrames)/12.5)s) "
            + "in \(String(format: "%.1f", -t0.timeIntervalSinceNow))s")

        // x-vector clone with the same say-clip the app run used.
        let (refSamples, refRate) = try AudioFileLoader.loadWAV(url: Self.refWAV)
        let t1 = Date()
        let cloned = model.synthesizeWithVoiceClone(
            text: Self.text, referenceAudio: refSamples,
            referenceSampleRate: refRate, sampling: sampling)
        let clFrames = cloned.count / 1920
        print("[E1-REPRO] x-vector:  \(clFrames) frames (~\(Double(clFrames)/12.5)s) "
            + "in \(String(format: "%.1f", -t1.timeIntervalSinceNow))s")
        print("[E1-REPRO] verdict: zero-shot \(zsFrames < 200 ? "STOPPED" : "RUNAWAY"), "
            + "clone \(clFrames < 200 ? "STOPPED" : "RUNAWAY (hit cap)")")
    }
}
