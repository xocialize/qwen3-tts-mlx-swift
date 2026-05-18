import Foundation
import MLX
import MLXNN

/// A linear layer that can operate in either full-precision (Linear) or quantized (QuantizedLinear) mode.
///
/// Usage:
/// ```swift
/// let layer = FlexibleLinear(256, 512, bias: false, quantized: config.useQuantization,
///                            groupSize: config.groupSize, bits: config.bits)
/// let output = layer(input) // works the same regardless of mode
/// ```
///
/// Weight loading: Use `CommonWeightLoader.applyFlexibleLinearWeights()` which auto-detects the mode.
public class FlexibleLinear: Module {
    /// Whether this layer uses quantization
    public let isQuantized: Bool

    // Only one of these is non-nil at a time. Using separate @ModuleInfo ensures
    // MLX's module graph includes the child for parameter traversal.
    @ModuleInfo(key: "quantized") var quantizedLayer: QuantizedLinear?
    @ModuleInfo(key: "linear") var linearLayer: Linear?

    /// Create based on quantization flag
    public init(_ inputDimensions: Int, _ outputDimensions: Int, bias: Bool = false,
                quantized: Bool, groupSize: Int = 64, bits: Int = 4) {
        self.isQuantized = quantized
        if quantized {
            self._quantizedLayer.wrappedValue = QuantizedLinear(
                inputDimensions, outputDimensions, bias: bias,
                groupSize: groupSize, bits: bits)
            self._linearLayer.wrappedValue = nil
        } else {
            self._quantizedLayer.wrappedValue = nil
            self._linearLayer.wrappedValue = Linear(inputDimensions, outputDimensions, bias: bias)
        }
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        if let q = quantizedLayer {
            return q(x)
        } else if let l = linearLayer {
            return l(x)
        } else {
            fatalError("FlexibleLinear: no inner layer initialized")
        }
    }
}

/// SwiGLU MLP that supports both full-precision and quantized linear layers.
public class FlexibleMLP: Module {
    @ModuleInfo public var gateProj: FlexibleLinear
    @ModuleInfo public var upProj: FlexibleLinear
    @ModuleInfo public var downProj: FlexibleLinear

    public let isQuantized: Bool

    public init(hiddenSize: Int, intermediateSize: Int, quantized: Bool = true,
                groupSize: Int = 64, bits: Int = 4) {
        self.isQuantized = quantized
        self._gateProj.wrappedValue = FlexibleLinear(
            hiddenSize, intermediateSize, bias: false,
            quantized: quantized, groupSize: groupSize, bits: bits)
        self._upProj.wrappedValue = FlexibleLinear(
            hiddenSize, intermediateSize, bias: false,
            quantized: quantized, groupSize: groupSize, bits: bits)
        self._downProj.wrappedValue = FlexibleLinear(
            intermediateSize, hiddenSize, bias: false,
            quantized: quantized, groupSize: groupSize, bits: bits)

        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gate = silu(gateProj(x))
        let up = upProj(x)
        return downProj(gate * up)
    }
}
