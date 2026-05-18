import Foundation
import MLX
import MLXNN
import AudioCommon

/// Weight loading for TTS components
public enum TTSWeightLoader {

    // MARK: - Combined Talker + Code Predictor (single file load)

    public static func loadTalkerAndCodePredictorWeights(
        talker: TalkerModel,
        codePredictor: CodePredictorModel,
        from directory: URL
    ) throws {
        let allWeights = try CommonWeightLoader.loadAllSafetensors(from: directory)
        print("Loaded \(allWeights.count) weights from safetensors")

        // Split into talker and code predictor weights
        var talkerWeights: [String: MLXArray] = [:]
        var cpWeights: [String: MLXArray] = [:]
        for (key, value) in allWeights {
            if key.hasPrefix("talker.code_predictor.") {
                let strippedKey = String(key.dropFirst("talker.code_predictor.".count))
                cpWeights[strippedKey] = value
            } else if key.hasPrefix("talker.") {
                let strippedKey = String(key.dropFirst("talker.".count))
                talkerWeights[strippedKey] = value
            }
        }

        applyTalkerWeights(to: talker, from: talkerWeights)
        applyCodePredictorWeights(to: codePredictor, from: cpWeights)
    }

    // MARK: - Talker

    private static func applyTalkerWeights(
        to talker: TalkerModel,
        from talkerWeights: [String: MLXArray]
    ) {
        print("Found \(talkerWeights.count) talker weights (quantized: \(talker.config.useQuantization))")

        // Codec embedding (float, not quantized)
        CommonWeightLoader.applyEmbeddingWeights(
            to: talker.codecEmbedding, prefix: "model.codec_embedding", from: talkerWeights)

        // Text embedding (float, not quantized)
        CommonWeightLoader.applyEmbeddingWeights(
            to: talker.textEmbedding, prefix: "model.text_embedding", from: talkerWeights)

        // Text projection MLP (safetensors keys use "linear_fc1"/"linear_fc2")
        CommonWeightLoader.applyFlexibleLinearWeights(
            to: talker.textProjection.fc1, prefix: "text_projection.linear_fc1", from: talkerWeights)
        CommonWeightLoader.applyFlexibleLinearWeights(
            to: talker.textProjection.fc2, prefix: "text_projection.linear_fc2", from: talkerWeights)

        // Codec head
        CommonWeightLoader.applyFlexibleLinearWeights(
            to: talker.codecHead, prefix: "codec_head", from: talkerWeights)

        // Final norm
        CommonWeightLoader.applyRMSNormWeights(
            to: talker.norm, prefix: "model.norm", from: talkerWeights)

        // Transformer layers
        for (i, layer) in talker.layers.enumerated() {
            let prefix = "model.layers.\(i)"
            applyTalkerLayerWeights(to: layer, prefix: prefix, from: talkerWeights)
        }

        print("Applied weights to Talker (\(talker.layers.count) layers)")
    }

    private static func applyTalkerLayerWeights(
        to layer: TalkerDecoderLayer,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        // Self attention
        CommonWeightLoader.applyFlexibleLinearWeights(
            to: layer.selfAttn.qProj, prefix: "\(prefix).self_attn.q_proj", from: weights)
        CommonWeightLoader.applyFlexibleLinearWeights(
            to: layer.selfAttn.kProj, prefix: "\(prefix).self_attn.k_proj", from: weights)
        CommonWeightLoader.applyFlexibleLinearWeights(
            to: layer.selfAttn.vProj, prefix: "\(prefix).self_attn.v_proj", from: weights)
        CommonWeightLoader.applyFlexibleLinearWeights(
            to: layer.selfAttn.oProj, prefix: "\(prefix).self_attn.o_proj", from: weights)

        // Q/K norms
        CommonWeightLoader.applyRMSNormWeights(
            to: layer.selfAttn.qNorm, prefix: "\(prefix).self_attn.q_norm", from: weights)
        CommonWeightLoader.applyRMSNormWeights(
            to: layer.selfAttn.kNorm, prefix: "\(prefix).self_attn.k_norm", from: weights)

        // Layer norms
        CommonWeightLoader.applyRMSNormWeights(
            to: layer.inputLayerNorm, prefix: "\(prefix).input_layernorm", from: weights)
        CommonWeightLoader.applyRMSNormWeights(
            to: layer.postAttentionLayerNorm, prefix: "\(prefix).post_attention_layernorm", from: weights)

        // MLP
        CommonWeightLoader.applyFlexibleMLPWeights(
            to: layer.mlp, prefix: "\(prefix).mlp", from: weights)
    }

    // MARK: - Code Predictor

    private static func applyCodePredictorWeights(
        to codePredictor: CodePredictorModel,
        from cpWeights: [String: MLXArray]
    ) {
        print("Found \(cpWeights.count) code predictor weights (quantized: \(codePredictor.config.useQuantization))")

        // Dimension projection (1.7B only: 2048→1024)
        if let proj = codePredictor.smallToMtpProjection {
            CommonWeightLoader.applyFlexibleLinearWeights(
                to: proj, prefix: "small_to_mtp_projection", from: cpWeights)
            print("  Loaded small_to_mtp_projection (\(codePredictor.config.inputDim ?? 0)→\(codePredictor.config.hiddenSize))")
        }

        // Codec embeddings (15 tables — safetensors uses singular "codec_embedding")
        for i in 0..<codePredictor.codecEmbeddings.count {
            CommonWeightLoader.applyEmbeddingWeights(
                to: codePredictor.codecEmbeddings[i],
                prefix: "model.codec_embedding.\(i)",
                from: cpWeights)
        }

        // Transformer layers
        for (i, layer) in codePredictor.layers.enumerated() {
            let prefix = "model.layers.\(i)"
            applyCodePredictorLayerWeights(to: layer, prefix: prefix, from: cpWeights)
        }

        // Norm
        CommonWeightLoader.applyRMSNormWeights(
            to: codePredictor.norm, prefix: "model.norm", from: cpWeights)

        // LM heads (15)
        for i in 0..<codePredictor.lmHeads.count {
            CommonWeightLoader.applyFlexibleLinearWeights(
                to: codePredictor.lmHeads[i],
                prefix: "lm_head.\(i)",
                from: cpWeights)
        }

        print("Applied weights to Code Predictor (\(codePredictor.layers.count) layers)")
    }

    private static func applyCodePredictorLayerWeights(
        to layer: CodePredictorDecoderLayer,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        CommonWeightLoader.applyFlexibleLinearWeights(
            to: layer.selfAttn.qProj, prefix: "\(prefix).self_attn.q_proj", from: weights)
        CommonWeightLoader.applyFlexibleLinearWeights(
            to: layer.selfAttn.kProj, prefix: "\(prefix).self_attn.k_proj", from: weights)
        CommonWeightLoader.applyFlexibleLinearWeights(
            to: layer.selfAttn.vProj, prefix: "\(prefix).self_attn.v_proj", from: weights)
        CommonWeightLoader.applyFlexibleLinearWeights(
            to: layer.selfAttn.oProj, prefix: "\(prefix).self_attn.o_proj", from: weights)

        CommonWeightLoader.applyRMSNormWeights(
            to: layer.selfAttn.qNorm, prefix: "\(prefix).self_attn.q_norm", from: weights)
        CommonWeightLoader.applyRMSNormWeights(
            to: layer.selfAttn.kNorm, prefix: "\(prefix).self_attn.k_norm", from: weights)

        CommonWeightLoader.applyRMSNormWeights(
            to: layer.inputLayerNorm, prefix: "\(prefix).input_layernorm", from: weights)
        CommonWeightLoader.applyRMSNormWeights(
            to: layer.postAttentionLayerNorm, prefix: "\(prefix).post_attention_layernorm", from: weights)

        CommonWeightLoader.applyFlexibleMLPWeights(
            to: layer.mlp, prefix: "\(prefix).mlp", from: weights)
    }

    // MARK: - Speech Tokenizer Decoder

    public static func loadSpeechTokenizerDecoderWeights(
        into decoder: SpeechTokenizerDecoder,
        from directory: URL
    ) throws {
        let allWeights = try CommonWeightLoader.loadAllSafetensors(from: directory)

        print("Found \(allWeights.count) speech tokenizer weights total")

        // Load RVQ codebook weights (decoder-side quantizer)
        loadRVQWeights(into: decoder.splitRVQ, from: allWeights)

        // Pre-conv: decoder.pre_conv.conv.{weight,bias}
        CommonWeightLoader.applyConv1dWeights(
            to: decoder.preConv.conv, prefix: "decoder.pre_conv.conv", from: allWeights, transpose: true)

        // Pre-transformer input/output projections
        CommonWeightLoader.applyLinearWeights(
            to: decoder.transformer.inputProj, prefix: "decoder.pre_transformer.input_proj", from: allWeights)
        CommonWeightLoader.applyLinearWeights(
            to: decoder.transformer.outputProj, prefix: "decoder.pre_transformer.output_proj", from: allWeights)

        // Pre-transformer layers
        for (i, layer) in decoder.transformer.layers.enumerated() {
            loadDecoderTransformerLayerWeights(to: layer, index: i, from: allWeights)
        }

        // Pre-transformer norm
        CommonWeightLoader.applyRMSNormWeights(
            to: decoder.transformer.norm, prefix: "decoder.pre_transformer.norm", from: allWeights)

        // Upsample stages: decoder.upsample.{0,1}
        // Stage 0: .0.conv (transposed conv) + .1 (ConvNeXt)
        CommonWeightLoader.applyConvTransposed1dWeights(
            to: decoder.preUpsample1.conv, prefix: "decoder.upsample.0.0.conv", from: allWeights, transpose: true)
        loadConvNeXtBlockWeights(to: decoder.preConvNeXt1, prefix: "decoder.upsample.0.1", from: allWeights)
        // Stage 1
        CommonWeightLoader.applyConvTransposed1dWeights(
            to: decoder.preUpsample2.conv, prefix: "decoder.upsample.1.0.conv", from: allWeights, transpose: true)
        loadConvNeXtBlockWeights(to: decoder.preConvNeXt2, prefix: "decoder.upsample.1.1", from: allWeights)

        // Decoder initial conv: decoder.decoder.0.conv
        CommonWeightLoader.applyConv1dWeights(
            to: decoder.inputConv.conv, prefix: "decoder.decoder.0.conv", from: allWeights, transpose: true)

        // Decoder blocks: decoder.decoder.{1,2,3,4}
        for (i, block) in decoder.decoderBlocks.enumerated() {
            loadDecoderBlockWeights(to: block, blockKey: "decoder.decoder.\(i + 1)", from: allWeights)
        }

        // Final snake: decoder.decoder.5
        loadSnakeBetaWeights(to: decoder.finalSnake, prefix: "decoder.decoder.5", from: allWeights)

        // Final conv: decoder.decoder.6.conv
        CommonWeightLoader.applyConv1dWeights(
            to: decoder.finalConv.conv, prefix: "decoder.decoder.6.conv", from: allWeights, transpose: true)

        print("Applied weights to Speech Tokenizer Decoder")
    }

    // MARK: - RVQ Weight Loading

    private static func loadRVQWeights(
        into splitRVQ: SplitResidualVectorQuantizer,
        from weights: [String: MLXArray]
    ) {
        // rvq_first: 1 semantic codebook
        loadQuantizerCodebook(
            into: splitRVQ.rvqFirst.quantizers[0].embedding,
            prefix: "decoder.quantizer.rvq_first.vq.layers.0._codebook",
            from: weights)
        // rvq_first output_proj (Conv1d 256->512, kernel=1)
        CommonWeightLoader.applyConv1dWeights(
            to: splitRVQ.rvqFirst.outputProj,
            prefix: "decoder.quantizer.rvq_first.output_proj",
            from: weights, transpose: true)

        // rvq_rest: 15 acoustic codebooks
        for i in 0..<splitRVQ.rvqRest.numQuantizers {
            loadQuantizerCodebook(
                into: splitRVQ.rvqRest.quantizers[i].embedding,
                prefix: "decoder.quantizer.rvq_rest.vq.layers.\(i)._codebook",
                from: weights)
        }
        // rvq_rest output_proj
        CommonWeightLoader.applyConv1dWeights(
            to: splitRVQ.rvqRest.outputProj,
            prefix: "decoder.quantizer.rvq_rest.output_proj",
            from: weights, transpose: true)
    }

    private static func loadQuantizerCodebook(
        into embedding: Embedding,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        // Check for pre-computed embeddings first
        if let embed = weights["\(prefix).embed"] {
            let params: [String: NestedItem<String, MLXArray>] = ["weight": .value(embed)]
            embedding.update(parameters: ModuleParameters(values: params))
            return
        }

        // Compute from cluster_usage + embedding_sum
        if let usage = weights["\(prefix).cluster_usage"],
           let embSum = weights["\(prefix).embedding_sum"] {
            let eps = MLXArray(Float(1e-7))
            let clampedUsage = maximum(usage, eps).expandedDimensions(axis: -1)
            let computed = embSum / clampedUsage
            let params: [String: NestedItem<String, MLXArray>] = ["weight": .value(computed)]
            embedding.update(parameters: ModuleParameters(values: params))
        }
    }

    // MARK: - Decoder Component Weight Loading

    private static func loadDecoderTransformerLayerWeights(
        to layer: DecoderTransformerLayer,
        index: Int,
        from weights: [String: MLXArray]
    ) {
        let prefix = "decoder.pre_transformer.layers.\(index)"

        // Attention projections
        CommonWeightLoader.applyLinearWeights(
            to: layer.selfAttn.qProj, prefix: "\(prefix).self_attn.q_proj", from: weights)
        CommonWeightLoader.applyLinearWeights(
            to: layer.selfAttn.kProj, prefix: "\(prefix).self_attn.k_proj", from: weights)
        CommonWeightLoader.applyLinearWeights(
            to: layer.selfAttn.vProj, prefix: "\(prefix).self_attn.v_proj", from: weights)
        CommonWeightLoader.applyLinearWeights(
            to: layer.selfAttn.oProj, prefix: "\(prefix).self_attn.o_proj", from: weights)

        // Layer norms
        CommonWeightLoader.applyRMSNormWeights(
            to: layer.norm1, prefix: "\(prefix).input_layernorm", from: weights)
        CommonWeightLoader.applyRMSNormWeights(
            to: layer.norm2, prefix: "\(prefix).post_attention_layernorm", from: weights)

        // SwiGLU MLP
        CommonWeightLoader.applyLinearWeights(
            to: layer.gateProj, prefix: "\(prefix).mlp.gate_proj", from: weights)
        CommonWeightLoader.applyLinearWeights(
            to: layer.upProj, prefix: "\(prefix).mlp.up_proj", from: weights)
        CommonWeightLoader.applyLinearWeights(
            to: layer.downProj, prefix: "\(prefix).mlp.down_proj", from: weights)

        // LayerScale for attention and MLP
        if let scale = weights["\(prefix).self_attn_layer_scale.scale"] {
            let params: [String: NestedItem<String, MLXArray>] = ["scale": .value(scale.reshaped([1, 1, -1]))]
            layer.attnLayerScale.update(parameters: ModuleParameters(values: params))
        }
        if let scale = weights["\(prefix).mlp_layer_scale.scale"] {
            let params: [String: NestedItem<String, MLXArray>] = ["scale": .value(scale.reshaped([1, 1, -1]))]
            layer.mlpLayerScale.update(parameters: ModuleParameters(values: params))
        }
    }

    private static func loadConvNeXtBlockWeights(
        to block: ConvNeXtBlock,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        CommonWeightLoader.applyConv1dWeights(
            to: block.dwConv.conv, prefix: "\(prefix).dwconv.conv", from: weights, transpose: true)
        CommonWeightLoader.applyLayerNormWeights(
            to: block.norm, prefix: "\(prefix).norm", from: weights)
        CommonWeightLoader.applyLinearWeights(
            to: block.pwConv1, prefix: "\(prefix).pwconv1", from: weights)
        CommonWeightLoader.applyLinearWeights(
            to: block.pwConv2, prefix: "\(prefix).pwconv2", from: weights)

        // LayerScale gamma
        if let scale = weights["\(prefix).gamma"] {
            let params: [String: NestedItem<String, MLXArray>] = ["scale": .value(scale.reshaped([1, 1, -1]))]
            block.layerScale.update(parameters: ModuleParameters(values: params))
        }
    }

    private static func loadSnakeBetaWeights(
        to snake: SnakeBeta,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        if let alpha = weights["\(prefix).alpha"] {
            let params: [String: NestedItem<String, MLXArray>] = ["alpha": .value(alpha.reshaped([1, 1, -1]))]
            snake.update(parameters: ModuleParameters(values: params))
        }
        if let beta = weights["\(prefix).beta"] {
            let params: [String: NestedItem<String, MLXArray>] = ["beta": .value(beta.reshaped([1, 1, -1]))]
            snake.update(parameters: ModuleParameters(values: params))
        }
    }

    // MARK: - Speaker Encoder

    public static func loadSpeakerEncoderWeights(
        into encoder: SpeakerEncoder,
        from directory: URL
    ) throws {
        let allWeights = try CommonWeightLoader.loadAllSafetensors(from: directory)

        // Filter speaker encoder weights and strip prefix
        var seWeights: [String: MLXArray] = [:]
        for (key, value) in allWeights {
            if key.hasPrefix("speaker_encoder.") {
                seWeights[String(key.dropFirst("speaker_encoder.".count))] = value
            }
        }
        print("Found \(seWeights.count) speaker encoder weights")

        guard !seWeights.isEmpty else {
            throw NSError(domain: "TTSWeightLoader", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "No speaker encoder weights found in safetensors"])
        }

        // Detect weight format from initial conv (kernel_size=5, inputChannels=128).
        // MLX Conv1d expects [out, kernel, in] → [512, 5, 128] (dim(2) == 128)
        // PyTorch Conv1d stores  [out, in, kernel] → [512, 128, 5] (dim(2) == 5)
        // Pre-converted MLX models (mlx-community 0.6B) are already in MLX layout,
        // but locally quantized models (1.7B-Base-4bit) retain PyTorch layout.
        let needsTranspose: Bool
        if let initialWeight = seWeights["blocks.0.conv.weight"] {
            needsTranspose = initialWeight.dim(2) != 128
        } else {
            needsTranspose = false
        }
        if needsTranspose {
            print("Speaker encoder weights are in PyTorch format — transposing [O,I,K] → [O,K,I]")
        }

        // Initial conv (blocks.0 = TimeDelayNetBlock wrapping Conv1d)
        // Weight key: blocks.0.conv.weight/bias → our initialConv (Conv1d)
        applySpeakerConv1dWeights(to: encoder.initialConv, prefix: "blocks.0.conv", from: seWeights, transpose: needsTranspose)

        // 3 ECAPA blocks (blocks.1, blocks.2, blocks.3)
        let ecapaBlocks = [encoder.block1, encoder.block2, encoder.block3]
        for (i, block) in ecapaBlocks.enumerated() {
            let blockPrefix = "blocks.\(i + 1)"
            // tdnn1, tdnn2 (TimeDelayNetBlock → .conv)
            applySpeakerConv1dWeights(to: block.tdnn1, prefix: "\(blockPrefix).tdnn1.conv", from: seWeights, transpose: needsTranspose)
            applySpeakerConv1dWeights(to: block.tdnn2, prefix: "\(blockPrefix).tdnn2.conv", from: seWeights, transpose: needsTranspose)
            // res2net_block.blocks.{0-6} (each is TimeDelayNetBlock → .conv)
            for j in 0..<block.res2netBlock.blocks.count {
                applySpeakerConv1dWeights(
                    to: block.res2netBlock.blocks[j],
                    prefix: "\(blockPrefix).res2net_block.blocks.\(j).conv",
                    from: seWeights, transpose: needsTranspose)
            }
            // se_block.conv1, se_block.conv2
            applySpeakerConv1dWeights(to: block.seBlock.conv1, prefix: "\(blockPrefix).se_block.conv1", from: seWeights, transpose: needsTranspose)
            applySpeakerConv1dWeights(to: block.seBlock.conv2, prefix: "\(blockPrefix).se_block.conv2", from: seWeights, transpose: needsTranspose)
        }

        // MFA (TimeDelayNetBlock → .conv)
        applySpeakerConv1dWeights(to: encoder.mfa.conv, prefix: "mfa.conv", from: seWeights, transpose: needsTranspose)

        // ASP: tdnn (TimeDelayNetBlock → .conv) and conv (direct Conv1d)
        applySpeakerConv1dWeights(to: encoder.asp.tdnn, prefix: "asp.tdnn.conv", from: seWeights, transpose: needsTranspose)
        applySpeakerConv1dWeights(to: encoder.asp.conv, prefix: "asp.conv", from: seWeights, transpose: needsTranspose)

        // FC (direct Conv1d)
        applySpeakerConv1dWeights(to: encoder.fc, prefix: "fc", from: seWeights, transpose: needsTranspose)

        print("Applied weights to Speaker Encoder")
    }

    /// Apply Conv1d weights from safetensors to a speaker encoder layer.
    ///
    /// - Parameter transpose: When `true`, transposes weight from PyTorch format
    ///   `[out, in, kernel]` to MLX format `[out, kernel, in]`.
    private static func applySpeakerConv1dWeights(
        to conv: Conv1d,
        prefix: String,
        from weights: [String: MLXArray],
        transpose: Bool = false
    ) {
        var params: [String: NestedItem<String, MLXArray>] = [:]
        if let w = weights["\(prefix).weight"] {
            let weight = transpose ? w.transposed(0, 2, 1) : w
            params["weight"] = .value(weight)
        }
        if let b = weights["\(prefix).bias"] {
            params["bias"] = .value(b)
        }
        if !params.isEmpty {
            conv.update(parameters: ModuleParameters(values: params))
        }
    }

    /// Load decoder block weights (SEANet style)
    /// Block key structure: decoder.decoder.{N}.block.0 = Snake, .block.1 = TransposedConv,
    /// .block.{2,3,4} = ResBlocks (each with act1, conv1, act2, conv2)
    private static func loadDecoderBlockWeights(
        to block: DecoderBlock,
        blockKey: String,
        from weights: [String: MLXArray]
    ) {
        // block.0 = Snake activation
        loadSnakeBetaWeights(to: block.snake, prefix: "\(blockKey).block.0", from: weights)

        // block.1 = Transposed conv (upsample)
        CommonWeightLoader.applyConvTransposed1dWeights(
            to: block.upsample.conv, prefix: "\(blockKey).block.1.conv", from: weights, transpose: true)

        // block.{2,3,4} = 3 residual units
        for (j, unit) in block.residualUnits.enumerated() {
            let resPrefix = "\(blockKey).block.\(j + 2)"
            loadSnakeBetaWeights(to: unit.snake1, prefix: "\(resPrefix).act1", from: weights)
            CommonWeightLoader.applyConv1dWeights(
                to: unit.conv1.conv, prefix: "\(resPrefix).conv1.conv", from: weights, transpose: true)
            loadSnakeBetaWeights(to: unit.snake2, prefix: "\(resPrefix).act2", from: weights)
            CommonWeightLoader.applyConv1dWeights(
                to: unit.conv2.conv, prefix: "\(resPrefix).conv2.conv", from: weights, transpose: true)
        }
    }

    // MARK: - Speech Tokenizer Encoder

    /// Load encoder weights from the speech_tokenizer directory.
    ///
    /// Weights are stored in `{modelDir}/speech_tokenizer/model.safetensors` with
    /// prefix `encoder.`. Conv1d weights are transposed from PyTorch [O,I,K] to MLX [O,K,I].
    public static func loadSpeechTokenizerEncoderWeights(
        into encoder: SpeechTokenizerEncoder,
        from directory: URL
    ) throws {
        let allWeights = try CommonWeightLoader.loadAllSafetensors(from: directory)

        // Filter encoder-only weights and strip "encoder." prefix
        var encWeights: [String: MLXArray] = [:]
        for (key, value) in allWeights {
            if key.hasPrefix("encoder.") {
                encWeights[String(key.dropFirst("encoder.".count))] = value
            }
        }
        print("Found \(encWeights.count) speech tokenizer encoder weights")

        guard !encWeights.isEmpty else {
            throw NSError(domain: "TTSWeightLoader", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "No encoder weights found in speech_tokenizer safetensors"])
        }

        // SEANet encoder
        loadEncoderSEANetWeights(into: encoder.seanet, from: encWeights)

        // Transformer
        loadEncoderTransformerWeights(into: encoder.transformer, from: encWeights)

        // Downsample
        CommonWeightLoader.applyConv1dWeights(
            to: encoder.downsample.conv.conv,
            prefix: "downsample.conv", from: encWeights, transpose: true)

        // Split RVQ quantizer
        loadEncoderQuantizerWeights(into: encoder.quantizer, from: encWeights)

        print("Applied weights to Speech Tokenizer Encoder")
    }

    // MARK: - Encoder SEANet Weight Loading

    /// Load SEANet encoder weights.
    ///
    /// Weight key structure: `encoder.layers.{i}.conv.{weight,bias}`
    /// where layers are indexed accounting for ELU activations (no weights):
    ///   - 0: init conv
    ///   - 1: resnet block (stage 0)
    ///   - 2: ELU (skip)
    ///   - 3: downsample conv (stage 0)
    ///   - 4: resnet block (stage 1)
    ///   - 5: ELU (skip)
    ///   - 6: downsample conv (stage 1)
    ///   - ...
    ///   - 13: ELU (skip)
    ///   - 14: final conv
    private static func loadEncoderSEANetWeights(
        into seanet: EncoderSEANet,
        from weights: [String: MLXArray]
    ) {
        // Layer 0: init conv
        CommonWeightLoader.applyConv1dWeights(
            to: seanet.initConv.conv,
            prefix: "encoder.layers.0.conv", from: weights, transpose: true)

        // Stages: each stage is [resnet_block, ELU, downsample_conv]
        // Weight indices: stage i → resnet at (1 + i*3), ELU at (2 + i*3), downsample at (3 + i*3)
        for (i, stage) in seanet.stages.enumerated() {
            let resBlockIdx = 1 + i * 3

            // Resnet block internal convs use "block.{1,3}" indexing
            // (block.0 and block.2 are ELU placeholders with no weights)
            for (j, resBlock) in stage.resBlocks.enumerated() {
                let blockPrefix = "encoder.layers.\(resBlockIdx + j)"
                CommonWeightLoader.applyConv1dWeights(
                    to: resBlock.conv1.conv,
                    prefix: "\(blockPrefix).block.1.conv", from: weights, transpose: true)
                CommonWeightLoader.applyConv1dWeights(
                    to: resBlock.conv2.conv,
                    prefix: "\(blockPrefix).block.3.conv", from: weights, transpose: true)
            }

            // Downsample conv (index after resnet blocks + ELU)
            let downIdx = resBlockIdx + stage.resBlocks.count + 1  // +1 for ELU
            CommonWeightLoader.applyConv1dWeights(
                to: stage.downsampleConv.conv,
                prefix: "encoder.layers.\(downIdx).conv", from: weights, transpose: true)
        }

        // Final conv: layer 14
        CommonWeightLoader.applyConv1dWeights(
            to: seanet.finalConv.conv,
            prefix: "encoder.layers.14.conv", from: weights, transpose: true)
    }

    // MARK: - Encoder Transformer Weight Loading

    private static func loadEncoderTransformerWeights(
        into transformer: EncoderTransformerBlock,
        from weights: [String: MLXArray]
    ) {
        for (i, layer) in transformer.layers.enumerated() {
            let prefix = "encoder_transformer.layers.\(i)"

            // Attention projections
            CommonWeightLoader.applyLinearWeights(
                to: layer.selfAttn.qProj, prefix: "\(prefix).self_attn.q_proj", from: weights)
            CommonWeightLoader.applyLinearWeights(
                to: layer.selfAttn.kProj, prefix: "\(prefix).self_attn.k_proj", from: weights)
            CommonWeightLoader.applyLinearWeights(
                to: layer.selfAttn.vProj, prefix: "\(prefix).self_attn.v_proj", from: weights)
            CommonWeightLoader.applyLinearWeights(
                to: layer.selfAttn.oProj, prefix: "\(prefix).self_attn.o_proj", from: weights)

            // Layer norms
            CommonWeightLoader.applyLayerNormWeights(
                to: layer.inputLayernorm, prefix: "\(prefix).input_layernorm", from: weights)
            CommonWeightLoader.applyLayerNormWeights(
                to: layer.postAttentionLayernorm, prefix: "\(prefix).post_attention_layernorm", from: weights)

            // GELU MLP
            CommonWeightLoader.applyLinearWeights(
                to: layer.mlp.fc1, prefix: "\(prefix).mlp.fc1", from: weights)
            CommonWeightLoader.applyLinearWeights(
                to: layer.mlp.fc2, prefix: "\(prefix).mlp.fc2", from: weights)

            // LayerScale
            if let scale = weights["\(prefix).self_attn_layer_scale.scale"] {
                let params: [String: NestedItem<String, MLXArray>] = ["scale": .value(scale.reshaped([1, 1, -1]))]
                layer.selfAttnLayerScale.update(parameters: ModuleParameters(values: params))
            }
            if let scale = weights["\(prefix).mlp_layer_scale.scale"] {
                let params: [String: NestedItem<String, MLXArray>] = ["scale": .value(scale.reshaped([1, 1, -1]))]
                layer.mlpLayerScale.update(parameters: ModuleParameters(values: params))
            }
        }
    }

    // MARK: - Encoder Quantizer Weight Loading

    private static func loadEncoderQuantizerWeights(
        into quantizer: EncoderSplitRVQ,
        from weights: [String: MLXArray]
    ) {
        // Semantic quantizer
        loadEncoderRVQWeights(
            into: quantizer.semanticQuantizer,
            prefix: "quantizer.semantic_residual_vector_quantizer",
            from: weights)

        // Acoustic quantizer
        loadEncoderRVQWeights(
            into: quantizer.acousticQuantizer,
            prefix: "quantizer.acoustic_residual_vector_quantizer",
            from: weights)
    }

    private static func loadEncoderRVQWeights(
        into rvq: EncoderResidualVQ,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        // Input/output projections (Conv1d)
        CommonWeightLoader.applyConv1dWeights(
            to: rvq.inputProj, prefix: "\(prefix).input_proj", from: weights, transpose: true)
        CommonWeightLoader.applyConv1dWeights(
            to: rvq.outputProj, prefix: "\(prefix).output_proj", from: weights, transpose: true)

        // Codebook layers
        for (i, layer) in rvq.layers.enumerated() {
            let cbPrefix = "\(prefix).layers.\(i).codebook"

            // Compute embeddings from embed_sum / max(cluster_usage, 1e-7)
            if let usage = weights["\(cbPrefix).cluster_usage"],
               let embSum = weights["\(cbPrefix).embed_sum"] {
                let eps = MLXArray(Float(1e-7))
                let clampedUsage = maximum(usage, eps).expandedDimensions(axis: -1)
                let computed = embSum / clampedUsage
                let params: [String: NestedItem<String, MLXArray>] = ["embed": .value(computed)]
                layer.codebook.update(parameters: ModuleParameters(values: params))
            } else if let embed = weights["\(cbPrefix).embed"] {
                let params: [String: NestedItem<String, MLXArray>] = ["embed": .value(embed)]
                layer.codebook.update(parameters: ModuleParameters(values: params))
            }
        }
    }
}
