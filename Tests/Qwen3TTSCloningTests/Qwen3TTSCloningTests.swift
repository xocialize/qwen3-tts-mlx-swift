import XCTest
import MLX
@testable import Qwen3TTSCloning

// NOTE: Tests that use MLX GPU operations (MLXRandom, MLX.save, eval) are
// skipped in SPM CLI test runs due to missing Metal library. They pass
// when run from Xcode or with the metallib bundle present.
// See anime-studio/CLAUDE.md "Metal library not found" known issue.

final class Qwen3TTSCloningTests: XCTestCase {

    // MARK: - VoiceClonePrompt

    func testVoiceClonePromptInit() throws {
        guard metalAvailable() else { throw XCTSkip("Metal required") }
        let prompt = VoiceClonePrompt(
            speakerEmbedding: .zeros([1024]),
            refCodes: .zeros([16, 10]),
            refTokenIds: [1, 2, 3],
            language: "english",
            referenceDuration: 2.5
        )
        XCTAssertEqual(prompt.refCodecFrames, 10)
        XCTAssertEqual(prompt.refTokenIds.count, 3)
        XCTAssertEqual(prompt.language, "english")
        XCTAssertEqual(prompt.referenceDuration, 2.5)
    }

    func testVoiceClonePromptShapes() throws {
        guard metalAvailable() else { throw XCTSkip("Metal required") }
        let prompt = VoiceClonePrompt(
            speakerEmbedding: .zeros([1024]),
            refCodes: .zeros([16, 53]),
            refTokenIds: Array(0..<20),
            language: "japanese",
            referenceDuration: 4.24
        )
        XCTAssertEqual(prompt.speakerEmbedding.shape, [1024])
        XCTAssertEqual(prompt.refCodes.shape, [16, 53])
        XCTAssertEqual(prompt.refCodecFrames, 53)
        XCTAssertEqual(prompt.refTokenIds.count, 20)
        XCTAssertEqual(prompt.language, "japanese")
        XCTAssertEqual(prompt.referenceDuration, 4.24, accuracy: 0.01)
    }

    func testVoiceClonePromptZeroFrames() throws {
        guard metalAvailable() else { throw XCTSkip("Metal required") }
        let prompt = VoiceClonePrompt(
            speakerEmbedding: .zeros([1024]),
            refCodes: .zeros([16, 0]),
            refTokenIds: [],
            language: "english",
            referenceDuration: 0.0
        )
        XCTAssertEqual(prompt.refCodecFrames, 0)
        XCTAssertTrue(prompt.refTokenIds.isEmpty)
    }

    // MARK: - VoiceCloningMode

    func testVoiceCloningModeEnum() {
        let icl = VoiceCloningMode.icl
        let xvec = VoiceCloningMode.xVectorOnly
        XCTAssertNotEqual(icl, xvec)
        XCTAssertEqual(icl, .icl)
        XCTAssertEqual(xvec, .xVectorOnly)
    }

    // MARK: - VoiceCloningError

    func testErrorDescriptions() {
        let errors: [VoiceCloningError] = [
            .emptyReferenceAudio,
            .emptyReferenceText,
            .emptyTargetText,
            .encoderWeightsNotLoaded,
            .invalidSampleRate(8000),
            .promptCreationFailed("test"),
            .synthesisFailedEOS,
            .serializationFailed("test"),
            .deserializationFailed("test"),
        ]
        for error in errors {
            XCTAssertNotNil(error.errorDescription)
            XCTAssertFalse(error.errorDescription!.isEmpty)
        }
    }

    func testInvalidSampleRateIncludesValue() {
        let error = VoiceCloningError.invalidSampleRate(8000)
        XCTAssertTrue(error.errorDescription!.contains("8000"))
    }

    // MARK: - PromptSerializer Round-Trip (requires Metal)

    func testPromptSerializationRoundTrip() throws {
        // MLX.save / loadArrays require Metal — skip if unavailable
        guard metalAvailable() else {
            throw XCTSkip("Metal library not available in SPM CLI")
        }

        let original = VoiceClonePrompt(
            speakerEmbedding: MLXRandom.normal([1024]),
            refCodes: MLXRandom.randInt(low: 0, high: 2048, [16, 25]),
            refTokenIds: [100, 200, 300, 400, 500],
            language: "japanese",
            referenceDuration: 3.14
        )

        let tempDir = FileManager.default.temporaryDirectory
        let url = tempDir.appendingPathComponent("test_prompt_\(UUID().uuidString).safetensors")
        defer {
            try? FileManager.default.removeItem(at: url)
            try? FileManager.default.removeItem(at: url.appendingPathExtension("json"))
        }

        try PromptSerializer.save(original, to: url)
        let loaded = try PromptSerializer.load(from: url)

        XCTAssertEqual(loaded.speakerEmbedding.shape, original.speakerEmbedding.shape)
        XCTAssertEqual(loaded.refCodes.shape, original.refCodes.shape)
        XCTAssertEqual(loaded.refTokenIds, original.refTokenIds)
        XCTAssertEqual(loaded.language, original.language)
        XCTAssertEqual(loaded.referenceDuration, original.referenceDuration, accuracy: 0.001)

        let spkDiff = abs(loaded.speakerEmbedding - original.speakerEmbedding).sum()
        eval(spkDiff)
        XCTAssertEqual(spkDiff.item(Float.self), 0.0, accuracy: 1e-6)
    }

    func testPromptSerializationCreatesFiles() throws {
        guard metalAvailable() else {
            throw XCTSkip("Metal library not available in SPM CLI")
        }

        let prompt = VoiceClonePrompt(
            speakerEmbedding: .zeros([1024]),
            refCodes: .zeros([16, 5]),
            refTokenIds: [1],
            language: "english",
            referenceDuration: 1.0
        )

        let tempDir = FileManager.default.temporaryDirectory
        let url = tempDir.appendingPathComponent("test_files_\(UUID().uuidString).safetensors")
        defer {
            try? FileManager.default.removeItem(at: url)
            try? FileManager.default.removeItem(at: url.appendingPathExtension("json"))
        }

        try PromptSerializer.save(prompt, to: url)
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.appendingPathExtension("json").path))
    }

    func testPromptDeserializationMissingFile() {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("nonexistent_\(UUID().uuidString).safetensors")
        XCTAssertThrowsError(try PromptSerializer.load(from: url)) { error in
            guard let cloningError = error as? VoiceCloningError else {
                XCTFail("Expected VoiceCloningError")
                return
            }
            if case .deserializationFailed = cloningError { /* pass */ }
            else { XCTFail("Expected deserializationFailed") }
        }
    }

    func testPromptSerializationMultipleLanguages() throws {
        guard metalAvailable() else {
            throw XCTSkip("Metal library not available in SPM CLI")
        }

        let languages = ["english", "japanese", "chinese", "korean", "german"]
        let tempDir = FileManager.default.temporaryDirectory

        for lang in languages {
            let prompt = VoiceClonePrompt(
                speakerEmbedding: .zeros([1024]),
                refCodes: .zeros([16, 3]),
                refTokenIds: [42],
                language: lang,
                referenceDuration: 1.0
            )
            let url = tempDir.appendingPathComponent("test_\(lang)_\(UUID().uuidString).safetensors")
            defer {
                try? FileManager.default.removeItem(at: url)
                try? FileManager.default.removeItem(at: url.appendingPathExtension("json"))
            }

            try PromptSerializer.save(prompt, to: url)
            let loaded = try PromptSerializer.load(from: url)
            XCTAssertEqual(loaded.language, lang)
        }
    }

    func testPromptWithLargeTokenList() throws {
        guard metalAvailable() else {
            throw XCTSkip("Metal library not available in SPM CLI")
        }

        let largeTokenIds = Array(0..<1000)
        let prompt = VoiceClonePrompt(
            speakerEmbedding: .zeros([1024]),
            refCodes: .zeros([16, 100]),
            refTokenIds: largeTokenIds,
            language: "english",
            referenceDuration: 8.0
        )

        let tempDir = FileManager.default.temporaryDirectory
        let url = tempDir.appendingPathComponent("test_large_\(UUID().uuidString).safetensors")
        defer {
            try? FileManager.default.removeItem(at: url)
            try? FileManager.default.removeItem(at: url.appendingPathExtension("json"))
        }

        try PromptSerializer.save(prompt, to: url)
        let loaded = try PromptSerializer.load(from: url)
        XCTAssertEqual(loaded.refTokenIds.count, 1000)
        XCTAssertEqual(loaded.refTokenIds, largeTokenIds)
    }

    // MARK: - Helpers

    /// Check if Metal/MLX GPU operations are available.
    private func metalAvailable() -> Bool {
        // Try a trivial MLX GPU operation; if Metal isn't available it will crash
        // so we check for the metallib bundle instead
        let bundlePath = Bundle.main.bundlePath
        let metallibPath = (bundlePath as NSString)
            .deletingLastPathComponent + "/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib"
        return FileManager.default.fileExists(atPath: metallibPath)
    }
}
