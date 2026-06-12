import XCTest
@testable import Qwen3TTS

/// E9 regression: the CV-based tail trim must (a) remove a genuine appended tonal hum,
/// (b) respect the trim budget, (c) leave modulated (speech-like) audio alone.
/// Pure CPU — no MLX arrays.
final class TonalTrimTests: XCTestCase {
    let sr = 24000
    let model = Qwen3TTSModel(config: .base06B)

    /// Amplitude-modulated noise ≈ speech (high CV).
    func speechLike(seconds: Double) -> [Float] {
        var rng = SystemRandomNumberGenerator()
        return (0..<Int(seconds * Double(sr))).map { i in
            let env = 0.05 + 0.15 * abs(sin(Float(i) / Float(sr) * 6.0))  // syllable-rate AM
            let noise = Float(Double.random(in: -1...1, using: &rng))
            return env * noise
        }
    }

    /// Constant sine = tonal hum (CV ≈ 0).
    func hum(seconds: Double) -> [Float] {
        (0..<Int(seconds * Double(sr))).map { i in
            0.1 * sin(2 * Float.pi * 440 * Float(i) / Float(sr))
        }
    }

    func testTrimsGenuineTonalTail() {
        let audio = speechLike(seconds: 3.0) + hum(seconds: 0.5)
        let trimmed = model.trimTonalTail(audio, sampleRate: sr)
        let removed = Double(audio.count - trimmed.count) / Double(sr)
        XCTAssertGreaterThan(removed, 0.2, "should remove most of the 0.5s hum")
        XCTAssertLessThanOrEqual(removed, 0.65, "should not exceed the budget + fade")
    }

    func testBudgetCapsLongHum() {
        let audio = speechLike(seconds: 3.0) + hum(seconds: 2.0)   // hum > budget
        let trimmed = model.trimTonalTail(audio, sampleRate: sr)
        let removed = Double(audio.count - trimmed.count) / Double(sr)
        XCTAssertLessThanOrEqual(removed, 0.80, "trim must respect min(0.6s, 15%) budget")
    }

    func testLeavesSpeechAlone() {
        let audio = speechLike(seconds: 4.0)
        let trimmed = model.trimTonalTail(audio, sampleRate: sr)
        let removed = Double(audio.count - trimmed.count) / Double(sr)
        XCTAssertLessThan(removed, 0.15, "speech-like audio must not be trimmed")
    }
}
