import Foundation
import MLX
import MLXNN
import MLXFast

// MARK: - Encoder Resnet Block

/// Residual block with channel compression and ELU activation.
///
/// Architecture:
///     input (dim=D)
///       ├→ ELU → CausalConv1d(D→D/compress, k, dilation) → ELU → CausalConv1d(D/compress→D, k=1)
///       └→ Identity (shortcut)
///       → sum → output
class EncoderResnetBlock: Module {
    @ModuleInfo var conv1: CausalConv1d
    @ModuleInfo var conv2: CausalConv1d

    init(dim: Int, kernelSize: Int = 3, dilation: Int = 1, compress: Int = 2) {
        let hiddenDim = dim / compress
        self._conv1.wrappedValue = CausalConv1d(
            inputChannels: dim, outputChannels: hiddenDim,
            kernelSize: kernelSize, dilation: dilation)
        self._conv2.wrappedValue = CausalConv1d(
            inputChannels: hiddenDim, outputChannels: dim,
            kernelSize: 1)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = elu(x)
        h = conv1(h)
        h = elu(h)
        h = conv2(h)
        return x + h
    }
}

// MARK: - Encoder SEANet Stage

/// One downsampling stage: resnet blocks + ELU + strided conv.
class EncoderSEANetStage: Module {
    @ModuleInfo var resBlocks: [EncoderResnetBlock]
    @ModuleInfo var downsampleConv: CausalConv1d

    init(inputChannels: Int, outputChannels: Int, ratio: Int,
         numResLayers: Int, kernelSize: Int, dilationBase: Int, compress: Int) {
        self._resBlocks.wrappedValue = (0..<numResLayers).map { j in
            let dilation = Int(pow(Double(dilationBase), Double(j)))
            return EncoderResnetBlock(
                dim: inputChannels, kernelSize: kernelSize,
                dilation: dilation, compress: compress)
        }
        self._downsampleConv.wrappedValue = CausalConv1d(
            inputChannels: inputChannels, outputChannels: outputChannels,
            kernelSize: ratio * 2, stride: ratio)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for block in resBlocks {
            h = block(h)
        }
        h = elu(h)
        h = downsampleConv(h)
        return h
    }
}

// MARK: - Encoder SEANet

/// SEANet convolutional encoder: 960× downsample from 24kHz to 25Hz.
///
/// Init conv → 4 downsample stages (ratios [4,5,6,8]) → ELU → final conv.
/// Channels: 1 → 64 → 128 → 256 → 512 → 1024 → 512.
class EncoderSEANet: Module {
    @ModuleInfo var initConv: CausalConv1d
    @ModuleInfo var stages: [EncoderSEANetStage]
    @ModuleInfo var finalConv: CausalConv1d

    init(config: SpeechTokenizerEncoderConfig) {
        let ratios = config.encodingRatios  // [4, 5, 6, 8]
        var channels = config.numFilters    // 64

        self._initConv.wrappedValue = CausalConv1d(
            inputChannels: config.audioChannels, outputChannels: channels,
            kernelSize: config.kernelSize)

        var stageList: [EncoderSEANetStage] = []
        for ratio in ratios {
            let nextChannels = channels * 2
            stageList.append(EncoderSEANetStage(
                inputChannels: channels, outputChannels: nextChannels,
                ratio: ratio, numResLayers: config.numResidualLayers,
                kernelSize: config.residualKernelSize,
                dilationBase: config.dilationBase, compress: config.compressFactor))
            channels = nextChannels
        }
        self._stages.wrappedValue = stageList

        // Final conv: 1024 → 512
        self._finalConv.wrappedValue = CausalConv1d(
            inputChannels: channels, outputChannels: config.dimension,
            kernelSize: config.lastKernelSize)

        super.init()
    }

    /// Input: [B, T, 1]. Output: [B, T/960, 512].
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = initConv(x)
        for stage in stages {
            h = stage(h)
        }
        h = elu(h)
        h = finalConv(h)
        return h
    }
}

// MARK: - Encoder Transformer Attention

/// Multi-head attention with sliding-window causal masking and RoPE.
class EncoderSlidingWindowAttention: Module {
    let numHeads: Int
    let headDim: Int
    let slidingWindow: Int

    @ModuleInfo var qProj: Linear
    @ModuleInfo var kProj: Linear
    @ModuleInfo var vProj: Linear
    @ModuleInfo var oProj: Linear

    init(config: SpeechTokenizerEncoderConfig) {
        self.numHeads = config.numHeads
        self.headDim = config.headDim
        self.slidingWindow = config.slidingWindow

        let hidden = config.dimension
        self._qProj.wrappedValue = Linear(hidden, hidden, bias: false)
        self._kProj.wrappedValue = Linear(hidden, hidden, bias: false)
        self._vProj.wrappedValue = Linear(hidden, hidden, bias: false)
        self._oProj.wrappedValue = Linear(hidden, hidden, bias: false)

        super.init()
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let (B, T, _) = (x.dim(0), x.dim(1), x.dim(2))

        var q = qProj(x).reshaped([B, T, numHeads, headDim])
        var k = kProj(x).reshaped([B, T, numHeads, headDim])
        let v = vProj(x).reshaped([B, T, numHeads, headDim])

        // Apply RoPE
        q = EncoderRoPE.apply(q, cos: cos, sin: sin)
        k = EncoderRoPE.apply(k, cos: cos, sin: sin)

        // Transpose to [B, H, T, D]
        let qT = q.transposed(0, 2, 1, 3)
        let kT = k.transposed(0, 2, 1, 3)
        let vT = v.transposed(0, 2, 1, 3)

        // Scaled dot-product attention with causal sliding-window mask
        let scale = 1.0 / Float(headDim).squareRoot()
        var scores = matmul(qT, kT.transposed(0, 1, 3, 2)) * MLXArray(scale)

        let mask = createCausalSlidingWindowMask(seqLen: T)
        scores = scores + mask

        let weights = softmax(scores, axis: -1)
        var output = matmul(weights, vT)

        output = output.transposed(0, 2, 1, 3).reshaped([B, T, numHeads * headDim])
        return oProj(output)
    }

    private func createCausalSlidingWindowMask(seqLen: Int) -> MLXArray {
        let positions = MLXArray(0..<Int32(seqLen))
        let rowPos = positions.reshaped([seqLen, 1])
        let colPos = positions.reshaped([1, seqLen])

        let causal = colPos .> rowPos
        let distance = rowPos - colPos
        let outsideWindow = distance .>= MLXArray(Int32(slidingWindow))

        let masked = logicalOr(causal, outsideWindow)
        let maskValues = which(masked, MLXArray(Float(-1e9)), MLXArray(Float(0)))
        return maskValues.reshaped([1, 1, seqLen, seqLen])
    }
}

// MARK: - Encoder Transformer MLP

/// GELU MLP: Linear → GELU → Linear.
class EncoderTransformerMLP: Module {
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear

    init(config: SpeechTokenizerEncoderConfig) {
        self._fc1.wrappedValue = Linear(config.dimension, config.intermediateSize, bias: false)
        self._fc2.wrappedValue = Linear(config.intermediateSize, config.dimension, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(gelu(fc1(x)))
    }
}

// MARK: - Encoder Transformer Layer

/// Pre-norm transformer layer with LayerScale.
class EncoderTransformerLayer: Module {
    @ModuleInfo var inputLayernorm: LayerNorm
    @ModuleInfo var selfAttn: EncoderSlidingWindowAttention
    @ModuleInfo var selfAttnLayerScale: LayerScale
    @ModuleInfo var postAttentionLayernorm: LayerNorm
    @ModuleInfo var mlp: EncoderTransformerMLP
    @ModuleInfo var mlpLayerScale: LayerScale

    init(config: SpeechTokenizerEncoderConfig) {
        self._inputLayernorm.wrappedValue = LayerNorm(dimensions: config.dimension, eps: config.normEps)
        self._selfAttn.wrappedValue = EncoderSlidingWindowAttention(config: config)
        self._selfAttnLayerScale.wrappedValue = LayerScale(channels: config.dimension, initValue: config.layerScaleInit)
        self._postAttentionLayernorm.wrappedValue = LayerNorm(dimensions: config.dimension, eps: config.normEps)
        self._mlp.wrappedValue = EncoderTransformerMLP(config: config)
        self._mlpLayerScale.wrappedValue = LayerScale(channels: config.dimension, initValue: config.layerScaleInit)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        var h = inputLayernorm(x)
        h = selfAttn(h, cos: cos, sin: sin)
        h = selfAttnLayerScale(h)
        var out = x + h

        h = postAttentionLayernorm(out)
        h = mlp(h)
        h = mlpLayerScale(h)
        out = out + h

        return out
    }
}

// MARK: - Encoder Transformer

/// 8-layer causal transformer bottleneck with pre-computed RoPE.
class EncoderTransformerBlock: Module {
    @ModuleInfo var layers: [EncoderTransformerLayer]
    let config: SpeechTokenizerEncoderConfig

    init(config: SpeechTokenizerEncoderConfig) {
        self.config = config
        self._layers.wrappedValue = (0..<config.numTransformerLayers).map { _ in
            EncoderTransformerLayer(config: config)
        }
        super.init()
    }

    /// Input: [B, T, 512]. Output: [B, T, 512].
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let seqLen = x.dim(1)
        let (cos, sin) = EncoderRoPE.frequencies(
            seqLen: seqLen, headDim: config.headDim,
            theta: config.ropeTheta, dtype: x.dtype)

        var h = x
        for layer in layers {
            h = layer(h, cos: cos, sin: sin)
        }
        return h
    }
}

// MARK: - Encoder RoPE

/// Rotary position embedding for the encoder transformer.
enum EncoderRoPE {
    static func frequencies(
        seqLen: Int, headDim: Int, theta: Float = 10000.0, dtype: DType = .float32
    ) -> (cos: MLXArray, sin: MLXArray) {
        let halfDim = headDim / 2
        let freqExponent = MLXArray(stride(from: Float(0), to: Float(halfDim), by: 1))
            * (-2.0 / Float(headDim))
        let freqs = pow(MLXArray(theta), freqExponent)
        let positions = MLXArray(stride(from: Float(0), to: Float(seqLen), by: 1))
        let angles = outer(positions, freqs)
        let fullAngles = concatenated([angles, angles], axis: -1)

        return (
            cos: MLX.cos(fullAngles).asType(dtype).reshaped([1, seqLen, 1, headDim]),
            sin: MLX.sin(fullAngles).asType(dtype).reshaped([1, seqLen, 1, headDim])
        )
    }

    static func apply(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let halfDim = x.dim(-1) / 2
        let x1 = x[0..., 0..., 0..., ..<halfDim]
        let x2 = x[0..., 0..., 0..., halfDim...]
        let rotated = concatenated([negative(x2), x1], axis: -1)
        return x * cos + rotated * sin
    }
}

// MARK: - Encoder Stride Downsample

/// Stride-2 downsampling: 25Hz → 12.5Hz via CausalConv1d(512→512, k=4, s=2).
class EncoderStrideDownsample: Module {
    @ModuleInfo var conv: CausalConv1d

    init(config: SpeechTokenizerEncoderConfig) {
        let stride = 2
        self._conv.wrappedValue = CausalConv1d(
            inputChannels: config.dimension, outputChannels: config.dimension,
            kernelSize: stride * 2, stride: stride, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv(x)
    }
}

// MARK: - Encoder Euclidean Codebook

/// L2 nearest-neighbor codebook for encoding.
///
/// Codebook is loaded from `embed_sum / max(cluster_usage, 1e-7)`.
class EncoderEuclideanCodebook: Module {
    @ParameterInfo var embed: MLXArray
    let size: Int
    let dim: Int

    init(size: Int, dim: Int) {
        self.size = size
        self.dim = dim
        self._embed.wrappedValue = MLXArray.zeros([size, dim])
        super.init()
    }

    /// Find nearest codebook entry for each position.
    /// - Parameter x: [B, T, D]
    /// - Returns: Codebook indices [B, T] (Int32)
    func encode(_ x: MLXArray) -> MLXArray {
        // ||x - e||² = ||x||² - 2<x,e> + ||e||²
        let xNormSq = x.square().sum(axis: -1, keepDims: true)
        let eNormSq = embed.square().sum(axis: -1)
        let dotProduct = matmul(x, embed.T)
        let distances = xNormSq - 2 * dotProduct + eNormSq
        return distances.argMin(axis: -1).asType(.int32)
    }

    /// Look up codebook vectors by index.
    /// - Parameter codes: [B, T]
    /// - Returns: [B, T, D]
    func lookup(_ codes: MLXArray) -> MLXArray {
        embed[codes]
    }
}

// MARK: - Encoder Vector Quantizer

/// Single-layer VQ with input/output projections.
class EncoderVectorQuantizer: Module {
    @ModuleInfo var codebook: EncoderEuclideanCodebook

    init(codebookSize: Int, codebookDim: Int) {
        self._codebook.wrappedValue = EncoderEuclideanCodebook(size: codebookSize, dim: codebookDim)
        super.init()
    }

    func encode(_ x: MLXArray) -> MLXArray {
        codebook.encode(x)
    }

    func lookup(_ codes: MLXArray) -> MLXArray {
        codebook.lookup(codes)
    }
}

// MARK: - Encoder Residual VQ

/// Multi-layer residual vector quantizer with input/output projections.
class EncoderResidualVQ: Module {
    @ModuleInfo var inputProj: Conv1d
    @ModuleInfo var outputProj: Conv1d
    @ModuleInfo var layers: [EncoderVectorQuantizer]

    init(inputDim: Int, codebookDim: Int, codebookSize: Int, numLayers: Int) {
        self._inputProj.wrappedValue = Conv1d(
            inputChannels: inputDim, outputChannels: codebookDim,
            kernelSize: 1, bias: false)
        self._outputProj.wrappedValue = Conv1d(
            inputChannels: codebookDim, outputChannels: inputDim,
            kernelSize: 1, bias: false)
        self._layers.wrappedValue = (0..<numLayers).map { _ in
            EncoderVectorQuantizer(codebookSize: codebookSize, codebookDim: codebookDim)
        }
        super.init()
    }

    /// Encode input and return codes + residual in input space.
    /// - Parameter x: [B, T, inputDim]
    /// - Returns: (codes [B, numLayers, T], residual [B, T, inputDim])
    func encode(_ x: MLXArray) -> (codes: MLXArray, residual: MLXArray) {
        let projected = inputProj(x)  // [B, T, codebookDim]

        var residual = projected
        var allCodes: [MLXArray] = []

        for layer in layers {
            let codes = layer.encode(residual)
            let quantized = layer.lookup(codes)
            residual = residual - quantized
            allCodes.append(codes.expandedDimensions(axis: 1))
        }

        let stackedCodes = concatenated(allCodes, axis: 1)
        let residualInInputSpace = x - outputProj(projected - residual)

        return (stackedCodes, residualInInputSpace)
    }
}

// MARK: - Encoder Split RVQ

/// Split quantizer: 1 semantic + 15 acoustic codebooks (uses first 15 of 32 acoustic layers).
class EncoderSplitRVQ: Module {
    @ModuleInfo var semanticQuantizer: EncoderResidualVQ
    @ModuleInfo var acousticQuantizer: EncoderResidualVQ
    let numSemanticQuantizers: Int
    let numValidQuantizers: Int

    init(config: SpeechTokenizerEncoderConfig) {
        self.numSemanticQuantizers = config.numSemanticQuantizers
        self.numValidQuantizers = config.numQuantizers

        self._semanticQuantizer.wrappedValue = EncoderResidualVQ(
            inputDim: config.dimension, codebookDim: config.codebookDim,
            codebookSize: config.codebookSize, numLayers: config.numSemanticQuantizers)
        // Acoustic: use numQuantizers-1 active layers (15), but allocate all 32
        // from the weight file. Only the first 15 are used during encoding.
        self._acousticQuantizer.wrappedValue = EncoderResidualVQ(
            inputDim: config.dimension, codebookDim: config.codebookDim,
            codebookSize: config.codebookSize,
            numLayers: config.numAcousticQuantizers)
        super.init()
    }

    /// Encode features to 16-codebook indices.
    /// - Parameter x: [B, T, 512]
    /// - Returns: [B, 16, T] (Int32)
    func encode(_ x: MLXArray) -> MLXArray {
        let (semanticCodes, semanticResidual) = semanticQuantizer.encode(x)

        // Only use first (numValidQuantizers - numSemanticQuantizers) = 15 acoustic layers
        let numAcousticActive = numValidQuantizers - numSemanticQuantizers

        let projected = acousticQuantizer.inputProj(semanticResidual)
        var residual = projected
        var acousticCodes: [MLXArray] = []

        for i in 0..<numAcousticActive {
            let layer = acousticQuantizer.layers[i]
            let codes = layer.encode(residual)
            let quantized = layer.lookup(codes)
            residual = residual - quantized
            acousticCodes.append(codes.expandedDimensions(axis: 1))
        }

        let stackedAcoustic = concatenated(acousticCodes, axis: 1)
        return concatenated([semanticCodes, stackedAcoustic], axis: 1)
    }
}

// MARK: - Speech Tokenizer Encoder

/// Encodes 24kHz audio to 16-codebook RVQ tokens at 12.5Hz.
///
/// Pipeline: SEANet (960× downsample) → Transformer (8 layers) → Stride downsample (2×) → Split RVQ.
/// Shares matched codebooks with `SpeechTokenizerDecoder` when loaded from the same weights.
public class SpeechTokenizerEncoder: Module {
    public let config: SpeechTokenizerEncoderConfig

    @ModuleInfo var seanet: EncoderSEANet
    @ModuleInfo var transformer: EncoderTransformerBlock
    @ModuleInfo var downsample: EncoderStrideDownsample
    @ModuleInfo var quantizer: EncoderSplitRVQ

    public init(config: SpeechTokenizerEncoderConfig = SpeechTokenizerEncoderConfig()) {
        self.config = config
        self._seanet.wrappedValue = EncoderSEANet(config: config)
        self._transformer.wrappedValue = EncoderTransformerBlock(config: config)
        self._downsample.wrappedValue = EncoderStrideDownsample(config: config)
        self._quantizer.wrappedValue = EncoderSplitRVQ(config: config)
        super.init()
    }

    /// Encode raw audio to 16-codebook RVQ tokens.
    ///
    /// - Parameter audio: Mono audio as `[T]` or `[1, T]` float array at 24kHz.
    /// - Returns: Codebook indices `[16, T_encoded]` where `T_encoded ≈ T / 1920`.
    public func encode(audio: MLXArray) -> MLXArray {
        var x = audio
        if x.ndim == 1 {
            x = x.reshaped([1, -1, 1])
        } else if x.ndim == 2 {
            x = x.expandedDimensions(axis: -1)
        }

        var features = seanet(x)
        features = transformer(features)
        features = downsample(features)
        let codes = quantizer.encode(features)

        if audio.ndim <= 2 {
            return codes.squeezed(axis: 0)
        }
        return codes
    }
}
