import Foundation

/// Errors that can occur during voice cloning operations.
public enum VoiceCloningError: Error, LocalizedError {
    case emptyReferenceAudio
    case emptyReferenceText
    case emptyTargetText
    case encoderWeightsNotLoaded
    case invalidSampleRate(Int)
    case promptCreationFailed(String)
    case synthesisFailedEOS
    case serializationFailed(String)
    case deserializationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .emptyReferenceAudio:
            return "Reference audio is empty"
        case .emptyReferenceText:
            return "Reference text transcript is empty"
        case .emptyTargetText:
            return "Target text for synthesis is empty"
        case .encoderWeightsNotLoaded:
            return "Codec encoder weights not loaded — model must include SpeechTokenizerEncoder"
        case .invalidSampleRate(let sr):
            return "Invalid sample rate: \(sr)"
        case .promptCreationFailed(let msg):
            return "Voice clone prompt creation failed: \(msg)"
        case .synthesisFailedEOS:
            return "Synthesis reached max tokens without EOS"
        case .serializationFailed(let msg):
            return "Prompt serialization failed: \(msg)"
        case .deserializationFailed(let msg):
            return "Prompt deserialization failed: \(msg)"
        }
    }
}
