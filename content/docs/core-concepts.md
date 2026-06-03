---
title: "Core Concepts"
lead: "Understand the three building blocks of every GoMLX program: the backend manager, computation graphs, and the context."
weight: 3
---

## Overview

GoMLX is built on three layered abstractions. Understanding them makes every other part of the library click:

1. **Backend** — the connection to a hardware backend (CPU, GPU, TPU)
2. **Graph** — a computation graph that you define as a pure Go function
3. **Store** — a store for scoped, named and typed model parameters (weights), as well as hyperparameters.

You can use just the backend and graph for mathematical computing, or add a `Store` to build trainable models.

---

## Backend

The `compute.Backend` connects your Go process to a hardware+software backend abstraction capable of executing our 
computations. Create one at program startup and reuse it everywhere:

<!-- sync_code: file=core_concepts/graph/main.go tag=backend -->
```go
import (
	"github.com/gomlx/compute"
	_ "github.com/gomlx/gomlx/backends/default" // Includes default backends.
)

	backend := compute.MustNew() // auto-selects best available backend
	fmt.Printf("Backend: %s\n", backend.Description())
```

The backend owns the device memory, compiles graphs to native code, and manages data transfer between host and device.
One backend per process is the typical pattern.

{{< callout type="note" >}}
`compute.New()` selects the best available backend in order: CUDA GPU → Metal (Apple) → CPU. 
To pin a specific backend, use the environment variable `GOMLX_BACKEND` or during construction
use the form  `compute.NewWithConfig("go")`.
{{< /callout >}}

{{< callout type="note" >}}
The following backends are implemented so far:

- **"go"**: Pure Go implementation: simple, very portable but slower. It works in WASM also (so it can be
  used in websites).
- **"xla"** (or "xla:cpu", "xla:cuda", "xla:tpu"): uses [Google's XLA](https://openxla.org/), the same backend used by
 TensorFlow, Jax and optionally by PyTorch.
- **darwinml**: (**experimental**) uses Apple Metal when possible.
{{< /callout >}}

---

## Computation Graphs

A **graph** is a pure function that describes a computation in terms of `*graph.Node` values. GoMLX traces this function once, compiles it to XLA HLO, and produces an executable that runs entirely on the device.

```go
// Define a graph function — just a Go function returning nodes
addFn := graph.Compile(manager, func(g *graph.Graph) *graph.Node {
    a := graph.Parameter(g, "a", shapes.Make(dtypes.Float32, 4))
    b := graph.Parameter(g, "b", shapes.Make(dtypes.Float32, 4))
    return graph.Add(a, b)
})

// Execute it — inputs move to device, result moves back
result := addFn.Call(tensorA, tensorB)
```

### Why graphs?

This design gives XLA visibility over the entire computation so it can apply aggressive optimizations: operator fusion, memory layout selection, and loop unrolling — automatically.

Your Go code never runs on the GPU. Only the *compiled graph* runs there. This is the same design used by JAX and TensorFlow's `tf.function`.

### Nodes are values, not tensors

Inside a graph function, `*graph.Node` represents a future value. You cannot inspect its contents during graph construction — only after calling `.Call()`. Operations on nodes describe the graph structure.

```go
// This is graph construction — no computation happens here
x := graph.Parameter(g, "x", shapes.Make(dtypes.Float32, 3))
y := graph.Mul(x, graph.Const(g, float32(2.0)))
z := graph.ReduceSum(y, 0) // sum all elements

// This executes the compiled graph on-device
result := compiledFn.Call(xTensor) // returns a *tensors.Tensor
fmt.Println(result.Value()) // []float32{...}
```

---

## Shapes and Dtypes

Every node has a **shape**: a list of dimension sizes, plus a dtype. GoMLX checks shape compatibility at graph construction time — mismatches are caught before any computation runs.

```go
// Shape: [batch, height, width, channels]
imgShape := shapes.Make(dtypes.Float32, 32, 224, 224, 3)

// Scalar
scalarShape := shapes.Make(dtypes.Float64) // no dimensions

// Check shape at construction
x := graph.Parameter(g, "x", imgShape)
fmt.Println(x.Shape()) // Float32[32, 224, 224, 3]
```

Common dtypes: `dtypes.Float32`, `dtypes.Float64`, `dtypes.Int32`, `dtypes.Int64`, `dtypes.Bool`.

---

## The Context

The **context** is a hierarchical store for model parameters. Think of it as the model's named weight dictionary, with Go's type safety built in.

```go
import "github.com/gomlx/gomlx/ml/context"

ctx := context.New()

// Inside a graph function, variables are created or retrieved by name
func denseLayer(ctx *context.Context, x *graph.Node, units int) *graph.Node {
    w := ctx.WithInitializer(initializers.GlorotUniform).
        VariableWithShape("weights", shapes.Make(dtypes.Float32, x.Shape().Dim(-1), units))
    b := ctx.VariableWithShape("bias", shapes.Make(dtypes.Float32, units))
    return graph.Add(graph.MatMul(x, w.ValueGraph(x.Graph())), b.ValueGraph(x.Graph()))
}
```

### Scoping

Use `ctx.In("name")` to create named sub-scopes, which keeps weight names unique across layers:

```go
x = denseLayer(ctx.In("layer1"), x, 128) // weights stored at "layer1/weights"
x = denseLayer(ctx.In("layer2"), x, 64)  // weights stored at "layer2/weights"
```

### Checkpointing

The context can serialize all its variables to disk and restore them:

```go
checkpointHandler := checkpoints.Build(ctx).Dir("/tmp/my-model").Done()
checkpointHandler.Save()   // saves all variables to disk
checkpointHandler.Restore() // restores from the latest checkpoint
```

---

## Putting it together

Here is the minimal skeleton of a trainable GoMLX program:

```go
func main() {
    // 1. Connect to hardware
    manager := backends.New()

    // 2. Create a context to hold weights
    ctx := context.New()

    // 3. Define your model as a graph function
    trainer := train.NewTrainer(manager, ctx, myModelFn,
        losses.SparseCategoricalCrossEntropyLogits,
        optimizers.Adam(),
    )

    // 4. Run the training loop
    loop := train.NewLoop(trainer)
    loop.RunSteps(trainDataset, 10_000)
}
```

Each of these pieces — manager, graph, context, trainer — is independently replaceable. You can swap the optimizer, the backend, or the loss function without touching the rest.
