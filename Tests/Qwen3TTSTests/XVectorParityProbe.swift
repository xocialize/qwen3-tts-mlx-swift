import XCTest
import MLX
@testable import Qwen3TTS
import AudioCommon

/// E8 parity probe: Swift SpeakerMel + SpeakerEncoder vs the PT reference dump
/// (/tmp/pt_xvector_samantha.npy via tools — compared offline in Python).
final class XVectorParityProbe: XCTestCase {
    func testDumpSwiftXVector() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QTTS_E8_XVEC"] == "1", "QTTS_E8_XVEC=1")

        let model = try await Qwen3TTSModel.fromPretrained(
            modelId: CloneRunawayRepro.model8bit,
            tokenizerModelId: CloneRunawayRepro.tokenizerDir)
        let (refSamples, refRate) = try AudioFileLoader.loadWAV(url: CloneRunawayRepro.refWAV)

        let mels = SpeakerMel.compute(audio: refSamples, sampleRate: refRate)
        eval(mels)
        let m32 = mels.asType(.float32)
        print("[E8-XVEC] swift mels:", mels.shape,
              "mean", m32.mean().item(Float.self), "std",
              sqrt(m32.variance().item(Float.self)))

        let emb = model.speakerEncoder(mels).squeezed()
        eval(emb)
        let e32 = emb.asType(.float32)
        let norm = sqrt(e32.square().sum()).item(Float.self)
        print("[E8-XVEC] swift x-vector:", emb.shape, "norm", norm)
        print("[E8-XVEC] first5:", (0..<5).map { e32[$0].item(Float.self) })

        try MLX.save(
            arrays: ["mels": m32, "xvector": e32],
            url: URL(fileURLWithPath: "/tmp/swift_xvector_samantha.safetensors"))
        print("[E8-XVEC] saved → /tmp/swift_xvector_samantha.safetensors")
    }
}
