---
title: "Core Concepts"
lead: "Understand the three building blocks of every GoMLX program: the backend manager, computation graphs, and the context."
weight: 3
---

## Overview

GoMLX is built on three layered abstractions. Understanding them makes every other part of the library click:

1. **Backend** — the connection to a hardware backend (CPU, GPU, TPU).
2. **Graph** — a computation graph that you define as a pure Go function.
3. **Tensor** — a concrete multi-dimensional array (or scalar) value, used as input and output when executing graphs. 
3. **Store** — a scoped storage for named and typed model parameters (weights), as well as hyperparameters of a model.

You can use just the backend and graph for mathematical computing, or add a `Store` to build trainable models.

---

## Backend

The `compute.Backend` connects your Go process to a hardware+software backend abstraction capable of executing our 
computations. Create one at program startup and reuse it everywhere:

<!-- sync_code: file=core_concepts/graph/main.go tag=cell1 -->
```go
import (
	"github.com/gomlx/compute"
	_ "github.com/gomlx/gomlx/backends/default" // Includes default backends.
)
```
<div align="right"><small><a href="https://github.com/gomlx/gomlx/blob/main/examples/gomlx.github.io/core-concepts/graph/main.go">(See source)</a></small></div>

Output:

<!-- sync_code: file=core_concepts/graph/main.go output_tag=cell1 -->
```
Backend: xla:cuda - PJRT "cuda" plugin (/home/janpf/.local/lib/go-xla/nvidia/pjrt_c_api_cuda_plugin.so) v0.100 [StableHLO] [1 device(s)]
```

The backend owns the device memory, compiles graphs to native code, and manages data transfer between host and device.
One backend per process is the typical pattern.

{{< callout type="note" >}}
`compute.New()` selects the best available backend in order: CUDA GPU → Metal (Apple) → CPU. 
To pin a specific backend, use the environment variable `GOMLX_BACKEND` or during construction
use the form  `compute.NewWithConfig("go")`.
{{< /callout >}}

The following backends are implemented so far:

- **"go"**: Pure Go implementation: simple, very portable but slower. It works in WASM also (so it can be
  used in websites).
- **"xla"** (or "xla:cpu", "xla:cuda", "xla:tpu"): uses [Google's XLA](https://openxla.org/), the same backend used by
 TensorFlow, Jax and optionally by PyTorch.
- [go-darwinml**](https://github.com/gomlx/go-darwinml): (**experimental, in development**) it provides
  the `CoreML` (ANE, GPU, CPU) and the `MPSGraph` (GPU/Metal) backends.

---

## Computation Graphs

A **graph** is a pure function that describes a computation in terms of `*graph.Node` values and operations
connecting them.
GoMLX provides the high-level API to build these graphs. 
The computation graphs are then JIT-compiled and can be executed very efficiently by the selected backend.

<!-- sync_code: file=core_concepts/graph/main.go tag=cell2 -->
```go
import (
	. "github.com/gomlx/gomlx/core/graph"
)

addFn := func(a, b *Node) *Node {
	fmt.Printf("* building addFn computation graph: a.shape=%s, b.shape=%s\n", a.Shape(), b.Shape())
	return Add(a, b)
}
addExec := MustNewExec1(backend, addFn)
fmt.Printf("\t- 1+1=%s\n", addExec.MustCall(1.0, 1.0))
fmt.Printf("\t- 2+2=%s\n", addExec.MustCall(2.0, 2.0))
```
<div align="right"><small><a href="https://github.com/gomlx/gomlx/blob/main/examples/gomlx.github.io/core-concepts/graph/main.go#L27">(See source)</a></small></div>

Output:

<!-- sync_code: file=core_concepts/graph/main.go output_tag=cell2 -->
```
* building addFn computation graph: a.shape=(Float64), b.shape=(Float64)
	- 1+1=float64(2)
	- 2+2=float64(4)
```

{{< callout type="note" >}}
- The `addFn` was called only once to build the graph -- hence the message "* building addFn" was only printed once. 
  After the graph was built and compiled, it was simply executed twice iwth `addExec.MustCall1()`. 
- We _dot imported_ the package `. "github.com/gomlx/gomlx/core/graph"`. This is common practice when most of the file contents are graph building blocks. 
{{< /callout >}}


### Why graphs?

This design gives the backend (XLA in this case) visibility over the entire computation so it can apply aggressive optimizations: operator fusion, memory layout selection, etc. — automatically.

Your Go code never runs on the GPU (or whatever is the accelerator). Only the *compiled graph* runs there. This is the same design used by JAX `@jax.jit` and TensorFlow's `@tf.function`.

### Nodes are "future values", not concrete tensors

Inside a graph function, a `*graph.Node` represents a future value. You cannot inspect its contents during graph construction — only after calling `.Call()` (or equivalent, like `.MustCall1`).
Operations on nodes describe the graph structure.

The `*graph.Node` does carry information about the shape (dimensions and data type) of the value though, and they are used during graph building to check compatibility 
of the nodes for the operations -- e.g.: adding an `int` to a `float`, or values with different ranks are not valid operations, and return
an error.

<!-- sync_code: file=core_concepts/graph/main.go tag=cell3 -->
```go
_, err := addExec.Call(int32(1), float32(1.0))
if err != nil {
	//...
}
```
<div align="right"><small><a href="https://github.com/gomlx/gomlx/blob/main/examples/gomlx.github.io/core-concepts/graph/main.go#L39">(See source)</a></small></div>

Output:

<!-- sync_code: file=core_concepts/graph/main.go output_tag=cell3 -->
```
* building addFn computation graph: a.shape=(Int32), b.shape=(Float32)
Error: cannot broadcast Int32 and Float32 for "Add": they have different dtypes
	.../gomlx.github.io/core-concepts/graph/main.go:29
	.../gomlx.github.io/core-concepts/graph/main.go:39
```

---

## Tensors


### Shapes and data types (dtypes)

Every node has a **shape**: a list of dimension sizes, plus a dtype -- and optionally names is using dynamic shapes. 
GoMLX checks shape compatibility at graph construction time — mismatches are caught before any computation runs (see example above).

Common dtypes: `dtypes.Float32`, `dtypes.Float64`, `dtypes.Int32`, `dtypes.Int64`, `dtypes.Bool`.

---

## The `model.Store`

The **store** is a hierarchical store for model parameters. Think of it as the model's named weight dictionary, with Go's type safety built in.

```go
import "github.com/gomlx/gomlx/ml/context"

ctx := context.New()

// Inside a graph function, variables are created or retrieved by name
func denseLayer(scope *context.Context, x *graph.Node, units int) *graph.Node {
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

---

## Training a Model

Here is the minimal skeleton of a trainable GoMLX program:

```go
func main() {
    // 1. Connect to hardware
    backend := compute.New()

    // 2. Create a store to hold weights
    store := model.New()

    // 3. Define your model as a graph function
    trainer := train.NewTrainer(backend, store, myModelFn,
        losses.SparseCategoricalCrossEntropyLogits,
        optimizers.Adam(),
    )

    // 4. Run the training loop
    loop := train.NewLoop(trainer)
    loop.RunSteps(trainDataset, 10_000)
}
```

Each of these pieces — backend, graph, store, trainer — is independently replaceable. You can swap the trainer, optimizer, the backend, or the loss function without touching the rest of .

Other related topics: **datasets**, **Hyperparameters**, **Losses**, **Optimizers**, **Metrics**.
