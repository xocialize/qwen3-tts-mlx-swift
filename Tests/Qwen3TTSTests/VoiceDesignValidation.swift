import XCTest
import MLX
@testable import Qwen3TTS
import AudioCommon

/// VoiceDesign validation (gated QTTS_VD=1): the flagship feature Qwen3-TTS was chosen for.
/// Checks per description: bounded termination, healthy amplitude (no tanh railing),
/// non-silence; writes WAVs to ~/Desktop for the listen check and for the offline
/// voice-distinctness analysis (PT speaker-encoder cosine matrix).
final class VoiceDesignValidation: XCTestCase {
    static let text = "The morning light spilled across the quiet harbor as the boats began to stir."

    struct VD { let id: String; let desc: String; let seed: UInt64 }
    static let voices: [VD] = [
        VD(id: "A1-deep-male",
           desc: "A deep, gravelly male voice with a slow, commanding tone.",
           seed: 101),
        VD(id: "A2-deep-male-reseed",
           desc: "A deep, gravelly male voice with a slow, commanding tone.",
           seed: 202),
        VD(id: "B-bright-female",
           desc: "A bright, cheerful young female voice, speaking quickly with high energy.",
           seed: 101),
        VD(id: "C-old-whisper",
           desc: "An elderly man speaking in a soft, breathy half-whisper, frail and slow.",
           seed: 101),
    ]

    func runSuite(modelDir: String, tag: String) async throws {
        let model = try await Qwen3TTSModel.fromPretrained(
            modelId: modelDir, tokenizerModelId: CloneRunawayRepro.tokenizerDir)

        for v in Self.voices {
            var sampling = SamplingConfig.voiceDesign
            sampling.seed = v.seed
            let t0 = Date()
            let audio = model.synthesizeVoiceDesign(
                text: Self.text, voiceDescription: v.desc, sampling: sampling)
            let dt = -t0.timeIntervalSinceNow
            let frames = audio.count / 1920
            let peak = audio.map { abs($0) }.max() ?? 0
            let rms = (audio.reduce(Float(0)) { $0 + $1 * $1 } / Float(max(audio.count, 1)))
                .squareRoot()
            print("[VD-\(tag)] \(v.id): \(frames) frames (~\(String(format: "%.1f", Double(frames)/12.5))s) "
                + "peak \(String(format: "%.3f", peak)) rms \(String(format: "%.3f", rms)) "
                + "in \(String(format: "%.1f", dt))s")

            XCTAssertGreaterThan(frames, 20, "\(v.id): implausibly short")
            XCTAssertLessThan(frames, 400, "\(v.id): runaway-class length")
            XCTAssertLessThan(peak, 0.98, "\(v.id): railing (E8-class saturation)")
            XCTAssertGreaterThan(rms, 0.01, "\(v.id): near-silence")
            XCTAssertLessThan(rms, 0.35, "\(v.id): hot output")

            try WAVWriter.write(
                samples: audio, sampleRate: 24000,
                to: URL(fileURLWithPath: NSHomeDirectory()
                    + "/Desktop/VD-\(tag)-\(v.id).wav"))
        }
    }

    func testVoiceDesignBF16() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QTTS_VD"] == "1", "QTTS_VD=1")
        try await runSuite(
            modelDir: "/Volumes/DEV_VOL1/anime-studio/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            tag: "bf16")
    }

    func testVoiceDesign4bit() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QTTS_VD4"] == "1", "QTTS_VD4=1")
        try await runSuite(
            modelDir: "/Volumes/DEV_VOL1/anime-studio/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit",
            tag: "4bit")
    }
}
