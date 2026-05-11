---
title: "GoMLX MNIST Tutorial: End-to-End Image Classification"
lead: "This tutorial walks through building, training, and evaluating both a Linear Model and a Convolutional Neural Network (CNN) on the classic MNIST dataset using GoMLX."
weight: 1
---

## The Big Picture: Core Concepts

Before writing the code, it helps to map out the primary systems interacting within a GoMLX pipeline. A standard machine learning task in GoMLX relies on four foundational pillars:

1. **`backends`**: The execution engine (typically XLA/StableHLO via PJRT) that compiles and runs the math on CPU, GPU, or TPU.
2. **`context`**: The state manager. It holds hyperparameters (like batch size and learning rate) and the learnable weights/variables of your model.
3. **`graph`**: The static computation definition. You build a directed acyclic graph of `Node`s representing mathematical operations.
4. **`train`**: The execution loop that orchestrates feeding data, calculating gradients, updating variables via an optimizer, and logging metrics.

## 1. Environment and Data Preparation

*Libraries used: `examples/mnist`, `backends`, `pkg/core/tensors/images`, `gonbui`*

To start, GoMLX uses the `backends` package to initialize the hardware execution environment. The `examples/mnist` package provides a highly convenient `NewDataset` function that downloads the data and wraps it in a data iterator.

```go
backend := backends.MustNew()
// mnist.NewDataset downloads the data and provides an iterator for the train loop
ds, err := mnist.NewDataset(backend, "Samples MNIST", dataDir, "train", dtypes.Float32)

```

*Note: When running interactively in GoNB, libraries like `gonbui` can be used to render HTML tables and inject images directly into the notebook cells to visualize the `images.ToImage()` outputs.*

## 2. Managing State with `context`

*Libraries used: `pkg/ml/context`, `ui/commandline`*

GoMLX uses `context.Context` to manage scopes, weights, and hyperparameters. Instead of hardcoding values like `batch_size` or `learning_rate` in the graph, you inject them into the context.

The `commandline` package allows you to seamlessly override these context parameters using flags (e.g., `-set="batch_size=17;model='cnn'"`).

```go
ctx := context.New()
ctx.SetParams(map[string]any{
    "model":           "cnn",
    "batch_size":      600,
    "learning_rate":   1e-4,
    "cnn_dropout_rate": 0.5,
})

```

## 3. Defining the Computation `graph`

*Libraries used: `pkg/core/graph`, `pkg/ml/layers`, `pkg/ml/layers/activations`*

The model function is where you define the mathematical operations. It takes the `context` (for weights/params) and a slice of input `Node`s.

### The Linear Model

A linear model simply reshapes the 28x28 image into a flat vector and passes it through a Dense layer.

```go
func LinearModelGraph(ctx *context.Context, spec any, inputs []*graph.Node) []*graph.Node {
    ctx = ctx.In("model") // Create a dedicated scope for the model's weights
    batchSize := inputs[0].Shape().Dimensions[0]
    
    // Flatten: [batch_size, 28, 28, 1] -> [batch_size, 784]
    embeddings := graph.Reshape(inputs[0], batchSize, -1)
    
    // Logits: [batch_size, 10]
    logits := layers.DenseWithBias(ctx, embeddings, mnist.NumClasses)
    return []*graph.Node{logits}
}

```

### The CNN Model

For a deeper network, the `layers` and `activations` packages provide the necessary building blocks:

* **`layers.Convolution`**: Creates feature maps.
* **`activations.Relu`**: Introduces non-linearity.
* **`graph.MaxPool`**: Downsamples spatial dimensions.
* **`layers.DropoutNormalize`**: Helps prevent overfitting during training.

```go
func CnnEmbeddings(ctx *context.Context, images *graph.Node) *graph.Node {
    batchSize := images.Shape().Dimensions[0]

    // Layer 1: Conv -> Relu -> MaxPool
    images = layers.Convolution(ctx.In("conv_1"), images).Filters(32).KernelSize(3).PadSame().Done()
    images = activations.Relu(images)
    images = graph.MaxPool(images).Window(2).Done()
    
    // Flatten and return
    return graph.Reshape(images, batchSize, -1)
}

```

## 4. The `train` Loop

*Libraries used: `pkg/ml/train`, `pkg/ml/train/optimizers`, `pkg/ml/train/losses`, `pkg/ml/train/metrics`*

Once the graph is defined, the `train.Loop` ties the dataset, context, and model graph together. GoMLX provides pre-built optimizers (like AdamW) and metrics (like Accuracy).

When `TrainModel` is invoked, it:

1. Re-loads parameters and weights from a checkpoint (if one exists).
2. Attaches progress bars and plotters (if `plots=true` is set in the context).
3. Executes the target number of training steps.
4. Evaluates against the test dataset.

```go
// Example of the output metric logging handled by the train loop:
// Results on test:
//   Mean Loss (#loss): 0.0356
//   Mean Accuracy (#acc): 98.75%

```
