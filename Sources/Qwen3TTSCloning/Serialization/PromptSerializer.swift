import Foundation
import MLX

/// Serializes and deserializes ``VoiceClonePrompt`` objects to disk.
///
/// Prompts are stored as a safetensors file (tensors) plus a JSON sidecar (metadata).
/// This enables one-time reference audio processing with unlimited reuse for synthesis.
public enum PromptSerializer {

    /// Save a voice clone prompt to disk.
    ///
    /// Creates two files:
    /// - `{url}` — safetensors with speakerEmbedding and refCodes
    /// - `{url}.json` — JSON with refTokenIds, refTextRoleTokenIds, language, referenceDuration
    public static func save(_ prompt: VoiceClonePrompt, to url: URL) throws {
        // Save tensors
        let tensors: [String: MLXArray] = [
            "speaker_embedding": prompt.speakerEmbedding,
            "ref_codes": prompt.refCodes,
        ]
        try MLX.save(arrays: tensors, url: url)

        // Save metadata as JSON sidecar
        let metadata = PromptMetadata(
            refTokenIds: prompt.refTokenIds,
            refTextRoleTokenIds: prompt.refTextRoleTokenIds,
            language: prompt.language,
            referenceDuration: prompt.referenceDuration
        )
        let jsonURL = url.appendingPathExtension("json")
        let jsonData = try JSONEncoder().encode(metadata)
        try jsonData.write(to: jsonURL)
    }

    /// Load a voice clone prompt from disk.
    ///
    /// Expects the same two files created by ``save(_:to:)``.
    /// Handles both v1 (without refTextRoleTokenIds) and v2 (with) metadata formats.
    public static func load(from url: URL) throws -> VoiceClonePrompt {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw VoiceCloningError.deserializationFailed("File not found: \(url.path)")
        }

        // Load tensors
        let tensors = try MLX.loadArrays(url: url)
        guard let speakerEmbedding = tensors["speaker_embedding"] else {
            throw VoiceCloningError.deserializationFailed("Missing speaker_embedding tensor")
        }
        guard let refCodes = tensors["ref_codes"] else {
            throw VoiceCloningError.deserializationFailed("Missing ref_codes tensor")
        }

        // Load metadata
        let jsonURL = url.appendingPathExtension("json")
        guard FileManager.default.fileExists(atPath: jsonURL.path) else {
            throw VoiceCloningError.deserializationFailed("Metadata file not found: \(jsonURL.path)")
        }
        let jsonData = try Data(contentsOf: jsonURL)
        let metadata = try JSONDecoder().decode(PromptMetadata.self, from: jsonData)

        print("[PromptSerializer] Loaded refCodes shape: \(refCodes.shape), speakerEmbedding shape: \(speakerEmbedding.shape)")

        if metadata.refTextRoleTokenIds.isEmpty {
            print("[PromptSerializer] WARNING: Loaded prompt has no refTextRoleTokenIds (v1 format).")
            print("  ICL mode will not have ChatML role wrapping for ref_text.")
            print("  Re-create the prompt with createPrompt() for best results.")
        }

        return VoiceClonePrompt(
            speakerEmbedding: speakerEmbedding,
            refCodes: refCodes,
            refTokenIds: metadata.refTokenIds,
            refTextRoleTokenIds: metadata.refTextRoleTokenIds,
            language: metadata.language,
            referenceDuration: metadata.referenceDuration
        )
    }
}

/// Internal metadata structure for JSON sidecar.
/// Supports both v1 (without refTextRoleTokenIds) and v2 (with) formats.
private struct PromptMetadata: Codable {
    let refTokenIds: [Int]
    let refTextRoleTokenIds: [Int]
    let language: String
    let referenceDuration: Float

    init(refTokenIds: [Int], refTextRoleTokenIds: [Int], language: String, referenceDuration: Float) {
        self.refTokenIds = refTokenIds
        self.refTextRoleTokenIds = refTextRoleTokenIds
        self.language = language
        self.referenceDuration = referenceDuration
    }

    /// Custom decoder to handle v1 metadata files that lack refTextRoleTokenIds.
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        refTokenIds = try container.decode([Int].self, forKey: .refTokenIds)
        refTextRoleTokenIds = try container.decodeIfPresent([Int].self, forKey: .refTextRoleTokenIds) ?? []
        language = try container.decode(String.self, forKey: .language)
        referenceDuration = try container.decode(Float.self, forKey: .referenceDuration)
    }

    private enum CodingKeys: String, CodingKey {
        case refTokenIds, refTextRoleTokenIds, language, referenceDuration
    }
}
