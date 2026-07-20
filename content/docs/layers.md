---
title: "Layers Library"
section: "Training & Layers"
weight: 27
---

GoMLX comes with a "batteries included" `ml/layers` library, providing a rich set of predefined, highly optimized neural network layers and architectures. 

Most layers use trainable variables (or "weights" or "model parameters"). Those are automatically created in the `model.Store`, at a given _scope_ passed to the layers (the `model.Scope` points to path withing a model's store). If you only have one layer within the current scope, you just pass the scope as is. Otherwise, usually one creates a sub-scope using `scope.In("MyLayer")` or something like `.In("MyLayer_%d", i)` if using multiple of the same type.

Most of these layers can be automatically configured using hyperparameters stored in the model's store (or a `model.Scope` pointing to one).This simplifies model definition and makes it easy to experiment with different layer types, activations, or normalizations by changing scope hyperparameters rather than changing Go code. 

Example:

```go
// FNN configured (num hidden layers, num hidden nodes, activation, etc.) by scope hyperparameters.
logits = fnn.New(scope.In("fnn"), logits, 1).Done()
```

{{% callout type="tip" %}}
Reading the implementations of these layers under `gomlx/ml/layers` is one of the best ways to learn how to write custom, complex vector computations and variable scopes in GoMLX.
{{% /callout %}}

---

## Core Layers

### Dense Layer
Adds a single fully connected linear transformation layer (with an optional bias term). It automatically applies weights regularization (L1/L2) from the scope, and supports multi-dimensional outputs.

```go
import "github.com/gomlx/gomlx/ml/layers"

// Dense layer with 128 output dimensions and a bias term
h := layers.Dense(scope.In("dense_0"), input, true, 128)

// Dense layer with multi-dimensional output shape [..., 64, 32]
h = layers.Dense(scope.In("dense_output"), input, true, 64, 32)
```

### Embeddings (`layers.Embedding`)
Transforms integer token indices into dense vector embeddings of a given dimension.

```go
// Embedding table of size 10,000 and dimension 128
embeddings := layers.Embedding(scope, tokenIds, dtypes.Float32, 10000, 128)
```

### Piecewise Linear Calibration 
Defined in `layers.PieceWiseLinearCalibration`, it maps numerical inputs into bounded calibrated spaces $[0, 1]$ using a set of keypoints (typically quantiles). 
This is useful for auto-calibration of inputs (such as continuous tabular features whose scale and distribution vary) and can also be used as learnable activation functions.

This example shows how the UCI-Adult example calibrates continuous columns:

```go
// Slice one column (feature) at a time from the continuous features tensor
featureSlice := Slice(continuousFeatures, AxisRange(), AxisRange(contIdx, contIdx+1))

// quantiles is a slice of float64 representing keypoints (e.g. from 0% to 100% quantiles)
quantiles := adultData.Quantiles[contIdx]

// Validate that quantiles are sorted and monotonically increasing
layers.AssertQuantilesForPWLCalibrationValid(quantiles)

// Calibrate the column values into [0, 1] based on the keypoints.
// Setting outputTrainable to true allows the calibrated outputs to adjust during training.
calibrated := layers.PieceWiseLinearCalibration(
	scope.In("continuous_calibration"),
	featureSlice,
	Const(g, quantiles),
	true, // outputTrainable
)
```

It's very flexible and supports various types of regularization and things like monotonicity enforcement.

---

## Activations

The `ml/layers/activation` package contains standard and advanced non-linear activation functions. It includes a helper to parse activations from the scope (e.g., `activation.ApplyFromScope(scope, x)`).

The complete list of supported activations includes:
* **`none`**: Identity / no-op.
* **`relu`**: Rectified Linear Unit ($Max(x, 0)$).
* **`sigmoid`**: Sigmoid function ($\frac{1}{1 + e^{-x}}$).
* **`hard_sigmoid`**: Piecewise linear approximation of sigmoid.
* **`leaky_relu`**: ReLU with a small slope for negative values.
* **`selu`**: Scaled Exponential Linear Unit, useful for self-normalizing networks.
* **`swish` / `silu`**: Sigmoid Linear Unit ($x \cdot \sigma(x)$).
* **`hard_swish`**: Faster, piecewise linear approximation of Swish.
* **`tanh`**: Hyperbolic Tangent activation.
* **`gelu`**: Gaussian Error Linear Unit.
* **`gelu_approx`** (alias **`gelu_pytorch_tanh`**): Highly efficient approximation of GELU.
* **`swiglu`**: Swish-Gated Linear Unit (commonly used in modern LLMs like LLaMA). It requires the input to have 2*hiddenDimension, half used as _value_, half as _gating_.

Examples:

```go
import "github.com/gomlx/gomlx/ml/layers/activation"

// RELU activation.
h = activation.Relu(h)

// Apply activation configured (hyperparameter) in current scope.
h = activation.ApplyFromScope(scope, h)
```

---

## Multi-Head Attention

The `ml/layers/attention` package provides a builder-based multi-head self and cross-attention layer. It is built to support modern Transformer features and hardware acceleration:

* **Grouped Query Attention (GQA)**: Supports sharding keys and values over fewer heads for efficient KV-cacheing.
* **Rotary Position Embeddings (RoPE)**: Direct frequency-based rotational encoding.
* **Attention Fusion**: Combines scores, scaling, masking, and softmax operations into fused GPU operations for high throughput.
* **Attention Masking**: Built-in causal, query, key, and pad masking.
* **Soft-Capping**: Prevents attention entropy collapse (Gemma 2 style).

```go
import "github.com/gomlx/gomlx/ml/layers/attention"

// Multi-Head Cross Attention with GQA, RoPE, and causal masking
attnOutput := attention.MultiHeadAttention(scope, query, key, value, numHeads, headDim).
    WithNumKVHeads(numKVHeads).
    WithRoPE(10000.0).
    WithCausalMask(true).
    WithFusion(true).
    Done()

// Convenient Self-Attention wrapper for 10 layers.
for i := range 10 {
    x = attention.SelfAttention(scope.In("attention_%d", i), x, numHeads, headDim).Done()
}
```

---

## Feed-Forward Networks

The `ml/layers/fnn` package implements Feed-Forward Neural Networks (or "MLPs" as in Multi-Layer Perceptrons) using a builder layout. 

Under the hood, `fnn` wraps `layers.Dense`, but provides the ability to automatically stack hidden layers. It can be fully configured via scope hyperparameters in the `model.Store` (such as `fnn.ParamNumHiddenLayers`, `fnn.ParamNumHiddenNodes`, `fnn.ParamResidual`, and `layers.ParamNormalization`).

* **Ensembling**: A unique feature is the ability to construct an ensemble of models directly in the computation graph. Ensembles share the batch axis and run in parallel, which is useful for bagging and uncertainty estimation.

```go
import "github.com/gomlx/gomlx/ml/layers/fnn"

// Standard MLP configured fluently
y := fnn.New(scope, input, outputDim).
    NumHiddenLayers(3, 256).
    Activation(activation.Swish).
    Normalization("layer").
    Dropout(0.1).
    Done()

// Ensemble of 5 MLPs running in parallel
ensembleY := fnn.New(scope, input, outputDim).
    NumHiddenLayers(2, 64).
    WithEnsembleSize(5).
    Done()

// Automatically configured FNN, all (num hidden layers, num hidden nodes, activation, etc.) configured by scope hyperparameters.
logits = fnn.New(scope.In("fnn"), logits, 1).Done()
```

---

## Kolmogorov-Arnold Networks

Kolmogorov-Arnold Networks (KANs) are an alternative to Multi-Layer Perceptrons (MLPs). Instead of having linear weights on nodes and static activations, KANs place learnable 1D functions on the edges.

* **Paper**: [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756)
* **B-splines KAN**: `.BSpline()` (the classical KAN).
* **Rational KAN**: `.Rational()` (approximates edge functions with learnable rational functions).
* **Discrete KAN**: `.Discrete()` (uses soft, differentiable piecewise-constant lookups, invented by the GoMLX authors).

```go
import "github.com/gomlx/gomlx/ml/layers/kan"

y := kan.New(scope, input, outputDim).
    NumHiddenLayers(1, 32).
    BSpline().
    Done()
```

---

## Rational Activations

Rational activations represent learnable activations of the form $f(x) = w \cdot \frac{P(x)}{Q(x)}$ where $P$ and $Q$ are polynomials. They can be initialized to approximate standard activations (like Swish or GELU) and then adaptively train.

* **Paper**: [Padé Activation Units: End-to-end Learning of Flexible Activation Functions](https://arxiv.org/abs/1907.06732)

```go
import "github.com/gomlx/gomlx/ml/layers/rational"

// Create a rational function initialized to approximate the Swish activation
actFn := rational.New(scope, input).
    Approximate("swish").
    Done()
```

---

## Normalization

The `ml/layers/norm` package handles normalizers. The helper `layers.MustNormalizeByName(scope, name, x)` selects normalizers via scope hyperparameters:

* **LayerNorm**: Standard layer normalization. Supports learned gains, offsets, and attention masking.
* **RMSNorm**: Root Mean Square Normalization. More computationally efficient because it omits centering (commonly used in LLMs).
* **BatchNorm**: Batch normalization with running averages. Includes `norm.UpdateBatchNormAverages(trainer, dataset)` to compute exact dataset statistics.
* **DynamicTanh**: An implementation of the paper [Transformers without Normalization](https://arxiv.org/abs/2503.10622). It serves as a normalization-free layer activation that stabilizes depth propagation by applying a bounded learnable scaling $tanh(\alpha \cdot x) \cdot \gamma + \beta$.

```go
import "github.com/gomlx/gomlx/ml/layers/norm"

h = norm.RMSNorm(scope, h).Done()
```

---

## Convolutions & DropBlock

### Convolution
Provides standard n-dimensional convolutions (Conv1D, Conv2D, Conv3D) based on input dimensions. It supports group convolutions (depthwise/group separable), strides, padding, and dilations.

```go
// 2D Convolution on an image tensor
conv := layers.Convolution(scope, images).
    Channels(64).
    KernelSize(3).
    PadSame().
    Done()
```

### DropBlock
A regularization technique for Convolutional Networks. Traditional dropout is ineffective in CNNs because neighboring pixels are highly correlated; DropBlock addresses this by dropping entire contiguous blocks of pixels.

* **Paper**: [DropBlock: A regularization method for convolutional networks](https://arxiv.org/abs/1810.12890)

```go
import "github.com/gomlx/gomlx/ml/layers"

// Apply DropBlock with block size of 5
h = layers.DropBlock(scope, h).
    WithBlockSize(5).
    WithDropoutProbability(0.1).
    Done()
```

---

## Recurrent Networks

The `ml/layers/lstm` package implements Long Short-Term Memory sequence models. It supports Peephole connections, bidirectional architectures, and ragged sequences (masking out padding tokens in variable-length batches).

```go
import "github.com/gomlx/gomlx/ml/layers/lstm"

// Bidirectional LSTM supporting ragged sequence padding
allStates, lastState, lastCell := lstm.New(scope, sequence, hiddenSize).
    Direction(lstm.Bidirectional).
    Ragged(sequenceLengths).
    Done()
```

---

## Vector Neural Networks

Vector Neural Networks (VNNs) extend neural networks to be equivariant to 3D rotations, meaning that rotating the inputs results in an equivalently rotated output. Tensors in VNNs have an extra axis of size 3 representing 3D coordinates.

* **Paper**: [Vector Neurons: A General Framework for SO(3)-Equivariant Networks](https://arxiv.org/abs/2104.12229)

```go
import "github.com/gomlx/gomlx/ml/layers/vnn"

// Create SO(3)-equivariant layers
y := vnn.New(scope, inputVectors, outputChannels).
    NumHiddenLayers(2, 64).
    Done()
```
