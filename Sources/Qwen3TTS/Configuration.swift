import Foundation

// MARK: - Talker Config

public struct TalkerConfig: Codable, Sendable {
    public var hiddenSize: Int = 1024
    public var numLayers: Int = 28
    public var numHeads: Int = 16
    public var numKVHeads: Int = 8
    public var headDim: Int = 128
    public var intermediateSize: Int = 3072
    public var ropeTheta: Float = 1_000_000.0
    public var mropeSections: [Int] = [24, 20, 20]
    public var rmsNormEps: Float = 1e-6
    public var textVocabSize: Int = 151936
    public var textHiddenSize: Int = 2048
    public var codecVocabSize: Int = 3072
    public var groupSize: Int = 64
    public var bits: Int = 4
    /// When false, uses full-precision Linear layers instead of QuantizedLinear.
    /// Auto-detected from config.json: false when no "quantization" key is present.
    public var useQuantization: Bool = true

    // Codec special token IDs (loaded from config.json, defaults match 0.6B)
    public var codecEosTokenId: Int = 2150
    public var codecPadId: Int = 2148
    public var codecBosId: Int = 2149
    public var codecThinkId: Int = 2154
    public var codecNothinkId: Int = 2155
    public var codecThinkBosId: Int = 2156
    public var codecThinkEosId: Int = 2157

    // Language ID mapping (loaded from config.json codec_language_id)
    public var codecLanguageIds: [String: Int]?

    // Speaker configuration (loaded from config.json spk_id / spk_is_dialect)
    public var speakerIds: [String: Int]?
    public var speakerDialects: [String: String]?

    public init() {}

    public static var base06B: TalkerConfig { TalkerConfig() }
}

// MARK: - Code Predictor Config

public struct CodePredictorConfig: Codable, Sendable {
    public var hiddenSize: Int = 1024
    public var numLayers: Int = 5
    public var numHeads: Int = 16
    public var numKVHeads: Int = 8
    public var headDim: Int = 128
    public var intermediateSize: Int = 3072
    public var ropeTheta: Float = 1_000_000.0
    public var rmsNormEps: Float = 1e-6
    public var vocabSize: Int = 2048
    public var numCodeGroups: Int = 16
    public var groupSize: Int = 64
    public var bits: Int = 4
    /// When false, uses full-precision Linear layers instead of QuantizedLinear.
    public var useQuantization: Bool = true
    /// Input embedding dimension (talker hidden size). When different from hiddenSize,
    /// a projection (small_to_mtp_projection) maps inputs from inputDim → hiddenSize.
    /// Codec embeddings also use this dimension. Defaults to hiddenSize.
    public var inputDim: Int?

    public init() {}
}

// MARK: - Speech Tokenizer Encoder Config

public struct SpeechTokenizerEncoderConfig: Codable, Sendable {
    public var dimension: Int = 512
    public var audioChannels: Int = 1
    public var numFilters: Int = 64
    public var ratios: [Int] = [8, 6, 5, 4]
    public var kernelSize: Int = 7
    public var lastKernelSize: Int = 3
    public var residualKernelSize: Int = 3
    public var numResidualLayers: Int = 1
    public var compressFactor: Int = 2
    public var dilationBase: Int = 2
    public var numTransformerLayers: Int = 8
    public var numHeads: Int = 8
    public var headDim: Int = 64
    public var intermediateSize: Int = 2048
    public var slidingWindow: Int = 250
    public var ropeTheta: Float = 10000.0
    public var layerScaleInit: Float = 0.01
    public var normEps: Float = 1e-5
    public var numQuantizers: Int = 16
    public var numSemanticQuantizers: Int = 1
    public var numAcousticQuantizers: Int = 32
    public var codebookSize: Int = 2048
    public var codebookDim: Int = 256
    public var sampleRate: Int = 24000

    public init() {}

    /// Encoding ratios (reversed for downsampling).
    public var encodingRatios: [Int] { ratios.reversed() }

    /// Total downsampling factor: product of ratios × stride-2 downsample = 1920.
    public var totalDownsampleFactor: Int {
        encodingRatios.reduce(1, *) * 2
    }
}

// MARK: - Speech Tokenizer Decoder Config

public struct SpeechTokenizerDecoderConfig: Codable, Sendable {
    public var latentDim: Int = 1024
    public var decoderDim: Int = 1536
    public var hiddenSize: Int = 512
    public var numHeads: Int = 16
    public var numKVHeads: Int = 16
    public var headDim: Int = 64
    public var numLayers: Int = 8
    public var upsampleRates: [Int] = [8, 5, 4, 3]
    public var upsamplingRatios: [Int] = [2, 2]
    public var numQuantizers: Int = 16
    public var semanticCodebookSize: Int = 2048
    public var acousticCodebookSize: Int = 2048
    public var codebookDim: Int = 256
    public var slidingWindow: Int = 72
    public var sampleRate: Int = 24000
    public var frameRate: Double = 12.5
    public var rmsNormEps: Float = 1e-8

    public init() {}
}

// MARK: - Special Codec Tokens

public struct CodecTokens {
    public static let codecPad: Int = 2148
    public static let codecBos: Int = 2149
    public static let codecEos: Int = 2150
    public static let codecThink: Int = 2154
    public static let codecNothink: Int = 2155
    public static let codecThinkBos: Int = 2156
    public static let codecThinkEos: Int = 2157
    public static let ttsPad: Int = 151671
    public static let ttsBos: Int = 151672
    public static let ttsEos: Int = 151673
    public static let languageEnglish: Int = 2050
    public static let languageGerman: Int = 2053
    public static let languageChinese: Int = 2055
    public static let languageJapanese: Int = 2058
    public static let languageSpanish: Int = 2054
    public static let languageFrench: Int = 2061
    public static let languageKorean: Int = 2064
    public static let languageRussian: Int = 2069
    public static let languageItalian: Int = 2070
    public static let languagePortuguese: Int = 2071
    public static let languageBeijingDialect: Int = 2074
    public static let languageSichuanDialect: Int = 2062

    public static func languageId(for language: String) -> Int? {
        switch language.lowercased() {
        case "english", "en": return languageEnglish
        case "german", "de": return languageGerman
        case "chinese", "zh": return languageChinese
        case "japanese", "ja": return languageJapanese
        case "spanish", "es": return languageSpanish
        case "french", "fr": return languageFrench
        case "korean", "ko": return languageKorean
        case "russian", "ru": return languageRussian
        case "italian", "it": return languageItalian
        case "portuguese", "pt": return languagePortuguese
        case "beijing_dialect": return languageBeijingDialect
        case "sichuan_dialect": return languageSichuanDialect
        default: return nil
        }
    }
}

// MARK: - Speaker Config

/// Parsed speaker data from CustomVoice model config.json
public struct SpeakerConfig: Sendable {
    /// Speaker name → codec token ID mapping
    public let speakerIds: [String: Int]
    /// Speaker name → dialect name mapping (e.g., "eric" → "sichuan_dialect")
    public let speakerDialects: [String: String]
    /// Dynamic language ID mapping from config.json codec_language_id
    public let codecLanguageIds: [String: Int]

    public var availableSpeakers: [String] { Array(speakerIds.keys).sorted() }

    public init(speakerIds: [String: Int], speakerDialects: [String: String], codecLanguageIds: [String: Int] = [:]) {
        self.speakerIds = speakerIds
        self.speakerDialects = speakerDialects
        self.codecLanguageIds = codecLanguageIds
    }
}

// MARK: - Streaming Config

/// Configuration for streaming TTS synthesis with chunked audio output.
public struct StreamingConfig: Sendable {
    /// Number of codec frames in the first emitted chunk (lower = lower latency).
    /// At 12.5 Hz, each frame = 80ms audio. Default 3 = 240ms audio.
    public var firstChunkFrames: Int

    /// Number of codec frames per subsequent chunk. Default 25 = 2s audio.
    public var chunkFrames: Int

    /// Left context frames for decoder quality (overlapping decode window). Default 10.
    public var decoderLeftContext: Int

    public init(firstChunkFrames: Int = 3, chunkFrames: Int = 25, decoderLeftContext: Int = 10) {
        self.firstChunkFrames = firstChunkFrames
        self.chunkFrames = chunkFrames
        self.decoderLeftContext = decoderLeftContext
    }

    /// Balanced defaults: ~225ms first-packet latency, 2s subsequent chunks.
    public static var `default`: StreamingConfig { .init() }

    /// Low-latency preset: ~120ms first-packet latency, smaller chunks.
    public static var lowLatency: StreamingConfig { .init(firstChunkFrames: 1, chunkFrames: 15) }
}

// MARK: - Model Variant

/// Well-known TTS model variants
public enum TTSModelVariant: String, CaseIterable, Sendable {
    // 0.6B variants (HuggingFace IDs — downloaded automatically)
    case base = "aufklarer/Qwen3-TTS-12Hz-0.6B-Base-MLX-4bit"
    case customVoice = "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit"
    // 1.7B variants — 4-bit quantized (folder names — resolved via ModelPathResolver.ttsModelDirectory)
    case base17B = "Qwen3-TTS-12Hz-1.7B-Base-4bit"
    case customVoice17B = "Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit"
    case voiceDesign17B = "Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit"
    // 1.7B variants — full precision (bfloat16 safetensors)
    case base17BFP = "Qwen3-TTS-12Hz-1.7B-Base"
    case customVoice17BFP = "Qwen3-TTS-12Hz-1.7B-CustomVoice"
    case voiceDesign17BFP = "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
}

// MARK: - Combined TTS Config

public struct Qwen3TTSConfig: Codable, Sendable {
    public var talker: TalkerConfig
    public var codePredictor: CodePredictorConfig
    public var speechTokenizerDecoder: SpeechTokenizerDecoderConfig

    /// Encoder config — populated when speech_tokenizer/config.json has encoder_config.
    /// Only Base models include encoder weights for voice cloning.
    public var speechTokenizerEncoder: SpeechTokenizerEncoderConfig?

    /// Model type from config.json: "base", "custom_voice", "voice_design"
    public var ttsModelType: String = "base"
    /// Model size from config.json: "0b6", "1b7"
    public var ttsModelSize: String = "0b6"
    /// Speaker encoder output dimension (1024 for 0.6B, 2048 for 1.7B)
    public var speakerEncoderDim: Int = 1024

    public init(
        talker: TalkerConfig = TalkerConfig(),
        codePredictor: CodePredictorConfig = CodePredictorConfig(),
        speechTokenizerDecoder: SpeechTokenizerDecoderConfig = SpeechTokenizerDecoderConfig()
    ) {
        self.talker = talker
        self.codePredictor = codePredictor
        self.speechTokenizerDecoder = speechTokenizerDecoder
    }

    public static var base06B: Qwen3TTSConfig {
        Qwen3TTSConfig()
    }

    /// Load config dynamically from a model directory's config.json.
    /// Populates TalkerConfig, CodePredictorConfig dimensions, codec tokens,
    /// speaker IDs, and model metadata from the JSON.
    public static func fromConfigJSON(at modelDir: URL) throws -> Qwen3TTSConfig {
        let configURL = modelDir.appendingPathComponent("config.json")
        let data = try Data(contentsOf: configURL)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NSError(domain: "Qwen3TTSConfig", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "config.json is not a valid JSON object"
            ])
        }

        var config = Qwen3TTSConfig()

        // Top-level model metadata
        if let modelType = root["tts_model_type"] as? String {
            config.ttsModelType = modelType
        }
        if let modelSize = root["tts_model_size"] as? String {
            config.ttsModelSize = modelSize
        }

        // Speaker encoder config
        if let sec = root["speaker_encoder_config"] as? [String: Any] {
            if let v = sec["enc_dim"] as? Int { config.speakerEncoderDim = v }
        }

        // Quantization info — auto-detect: no "quantization" key means full-precision
        if let quant = root["quantization"] as? [String: Any] {
            config.talker.useQuantization = true
            config.codePredictor.useQuantization = true
            if let groupSize = quant["group_size"] as? Int {
                config.talker.groupSize = groupSize
                config.codePredictor.groupSize = groupSize
            }
            if let bits = quant["bits"] as? Int {
                config.talker.bits = bits
                config.codePredictor.bits = bits
            }
        } else {
            // No quantization key → full-precision model
            config.talker.useQuantization = false
            config.codePredictor.useQuantization = false
        }

        // Talker config
        if let tc = root["talker_config"] as? [String: Any] {
            if let v = tc["hidden_size"] as? Int { config.talker.hiddenSize = v }
            if let v = tc["intermediate_size"] as? Int { config.talker.intermediateSize = v }
            if let v = tc["num_hidden_layers"] as? Int { config.talker.numLayers = v }
            if let v = tc["num_attention_heads"] as? Int { config.talker.numHeads = v }
            if let v = tc["num_key_value_heads"] as? Int { config.talker.numKVHeads = v }
            if let v = tc["head_dim"] as? Int { config.talker.headDim = v }
            if let v = tc["text_hidden_size"] as? Int { config.talker.textHiddenSize = v }
            if let v = tc["text_vocab_size"] as? Int { config.talker.textVocabSize = v }
            if let v = tc["vocab_size"] as? Int { config.talker.codecVocabSize = v }
            if let v = tc["rope_theta"] as? Double { config.talker.ropeTheta = Float(v) }
            if let v = tc["rms_norm_eps"] as? Double { config.talker.rmsNormEps = Float(v) }
            if let rs = tc["rope_scaling"] as? [String: Any],
               let sections = rs["mrope_section"] as? [Int] {
                config.talker.mropeSections = sections
            }

            // Codec special token IDs
            if let v = tc["codec_eos_token_id"] as? Int { config.talker.codecEosTokenId = v }
            if let v = tc["codec_pad_id"] as? Int { config.talker.codecPadId = v }
            if let v = tc["codec_bos_id"] as? Int { config.talker.codecBosId = v }
            if let v = tc["codec_think_id"] as? Int { config.talker.codecThinkId = v }
            if let v = tc["codec_nothink_id"] as? Int { config.talker.codecNothinkId = v }
            if let v = tc["codec_think_bos_id"] as? Int { config.talker.codecThinkBosId = v }
            if let v = tc["codec_think_eos_id"] as? Int { config.talker.codecThinkEosId = v }

            // Language ID mapping
            if let langMap = tc["codec_language_id"] as? [String: Int] {
                config.talker.codecLanguageIds = langMap
            }

            // Speaker configuration
            if let spkMap = tc["spk_id"] as? [String: Int], !spkMap.isEmpty {
                config.talker.speakerIds = spkMap
            }
            if let dialectMap = tc["spk_is_dialect"] as? [String: Any], !dialectMap.isEmpty {
                // spk_is_dialect values are either false (Bool) or a dialect string
                var dialects: [String: String] = [:]
                for (speaker, value) in dialectMap {
                    if let dialectName = value as? String {
                        dialects[speaker] = dialectName
                    }
                    // false/Bool values → speaker has no dialect, skip
                }
                if !dialects.isEmpty {
                    config.talker.speakerDialects = dialects
                }
            }

            // Code predictor config (nested inside talker_config)
            if let cpc = tc["code_predictor_config"] as? [String: Any] {
                if let v = cpc["hidden_size"] as? Int { config.codePredictor.hiddenSize = v }
                if let v = cpc["intermediate_size"] as? Int { config.codePredictor.intermediateSize = v }
                if let v = cpc["num_hidden_layers"] as? Int { config.codePredictor.numLayers = v }
                if let v = cpc["num_attention_heads"] as? Int { config.codePredictor.numHeads = v }
                if let v = cpc["num_key_value_heads"] as? Int { config.codePredictor.numKVHeads = v }
                if let v = cpc["head_dim"] as? Int { config.codePredictor.headDim = v }
                if let v = cpc["vocab_size"] as? Int { config.codePredictor.vocabSize = v }
                if let v = cpc["num_code_groups"] as? Int { config.codePredictor.numCodeGroups = v }
                if let v = cpc["rope_theta"] as? Double { config.codePredictor.ropeTheta = Float(v) }
                if let v = cpc["rms_norm_eps"] as? Double { config.codePredictor.rmsNormEps = Float(v) }
            }

            // Set inputDim when talker and code predictor have different hidden sizes
            // (e.g., 1.7B: talker=2048, CP=1024 → needs small_to_mtp_projection)
            if config.talker.hiddenSize != config.codePredictor.hiddenSize {
                config.codePredictor.inputDim = config.talker.hiddenSize
            }
        }

        return config
    }
}
