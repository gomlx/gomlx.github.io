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

Tensors hold the inputs and outputs of a graph computation. They represent concrete multi-dimensional array values (from scalar 0D to arbitrary dimensions), defined by their **shape** (which specifies the dimensions and a data type, or `DType`).

### Shapes and data types (dtypes)

Every tensor has a shape (e.g. `(Float32)[2, 2]`), a list of dimension sizes, plus a `DType`. GoMLX checks shape compatibility at graph construction time, catching mismatches before any computation starts.

You can construct [Tensor](file:///home/janpf/Projects/gomlx/gomlx/core/tensors/tensor.go) objects from standard Go values (like multi-dimensional slices) using [FromValue](file:///home/janpf/Projects/gomlx/gomlx/core/tensors/local.go#L799):

<!-- sync_code: file=core_concepts/tensors/main.go tag=create -->
```go
// Tensors can be created from Go values, such as multi-dimensional slices
t := tensors.FromValue([][]float32{{1.0, 2.0}, {3.0, 4.0}})
fmt.Printf("Tensor shape: %s\n", t.Shape())
fmt.Printf("Tensor Go value: %v\n", t.Value())
```
<div align="right"><small><a href="https://github.com/gomlx/gomlx/blob/main/examples/gomlx.github.io/core-concepts/tensors/main.go#L20">(See source)</a></small></div>

Output:

<!-- sync_code: file=core_concepts/tensors/main.go output_tag=create -->
```
Tensor shape: (Float32)[2, 2]
Tensor Go value: [[1 2] [3 4]]
```

Common dtypes include `dtypes.Float32`, `dtypes.Float64`, `dtypes.Int32`, `dtypes.Int64`, and `dtypes.Bool`.

### Host vs. Device memory

Behind the scenes, a [Tensor](file:///home/janpf/Projects/gomlx/gomlx/core/tensors/tensor.go) maintains synchronization between its memory representation in host RAM (local CPU) and accelerator device memory (GPU/TPU):

<!-- sync_code: file=core_concepts/tensors/main.go tag=sync -->
```go
// Tensors cache data both locally (host CPU) and on accelerator devices.
// Transferring data between CPU and devices has a cost and is done lazily.
fmt.Printf("Has local copy? %v\n", t.HasLocal())
```
<div align="right"><small><a href="https://github.com/gomlx/gomlx/blob/main/examples/gomlx.github.io/core-concepts/tensors/main.go#L31">(See source)</a></small></div>

Output:

<!-- sync_code: file=core_concepts/tensors/main.go output_tag=sync -->
```
Has local copy? true
```

To avoid expensive memory copies, transferring data between host and device is performed lazily only when needed.

### Finalizing Tensors

Since the Go Garbage Collector cannot see memory allocated on accelerator devices (like CUDA GPUs), accelerator device memory is best if explicitly managed. 
When you are done with a tensor, you should explicitly call `FinalizeAll` (or `MustFinalizeAll`) to free its device buffers -- the GC will also free
the memory, but it may hold to it too long.

<!-- sync_code: file=core_concepts/tensors/main.go tag=finalize -->
```go
// Tensors allocate memory on accelerator devices (GPU, TPU).
// Because the Go Garbage Collector cannot track device memory,
// you must finalize tensors that are no longer in use to prevent memory leaks.
t.MustFinalizeAll()
```
<div align="right"><small><a href="https://github.com/gomlx/gomlx/blob/main/examples/gomlx.github.io/core-concepts/tensors/main.go#L41">(See source)</a></small></div>

### Images

The `github.com/gomlx/gomlx/core/tensors/images` package provides utilities to load standard Go images into tensors and export them back. When loading image batches, the resulting tensor shape is `[batch_size, height, width, channels]`:

<!-- sync_code: file=core_concepts/tensors/main.go tag=image -->
```go
// Create two simple blank images (e.g. 100x100 RGB).
img1 := image.NewRGBA(image.Rect(0, 0, 100, 100))
img2 := image.NewRGBA(image.Rect(0, 0, 100, 100))

// Load the batch of images into a Float32 tensor.
// The resulting shape is [batch_size, height, width, channels].
imagesTensor := timages.ToTensor(dtypes.Float32).Batch([]image.Image{img1, img2})
fmt.Printf("Batch images shape: %s\n", imagesTensor.Shape())
```
<div align="right"><small><a href="https://github.com/gomlx/gomlx/blob/main/examples/gomlx.github.io/core-concepts/tensors/main.go#L49">(See source)</a></small></div>

Output:

<!-- sync_code: file=core_concepts/tensors/main.go output_tag=image -->
```
Batch images shape: (Float32)[2, 100, 100, 3]
```

---

## The `model.Store` and Scopes

To build trainable models, you need a way to declare, retrieve, and update parameters (weights and biases) that persist across graph executions. The [Store](file:///home/janpf/Projects/gomlx/gomlx/ml/model/store.go#L33) is a hierarchical (tree-like) store for model parameters (represented by [Variable](file:///home/janpf/Projects/gomlx/gomlx/ml/model/variable.go)) and hyperparameters.

### Model Variables and Executors

Instead of using the basic `graph.Exec`, neural network architectures use [Exec](file:///home/janpf/Projects/gomlx/gomlx/ml/model/exec.go). It is constructed with a [Store](file:///home/janpf/Projects/gomlx/gomlx/ml/model/store.go#L33) and automatically handles passing variables as extra inputs and outputs to the compiled graph.

Here is a simple counter that increments a variable in the store on each step:

<!-- sync_code: file=core_concepts/store/main.go tag=counter -->
```go
store := model.NewStore()
counterFn := func(ctx *model.Scope, g *Graph) *Node {
	counterVar := ctx.VariableWithValue("counter", int32(0))
	counter := AddScalar(counterVar.NodeValue(g), 1)
	counterVar.SetNodeValue(counter)
	return counter
}

exec := model.MustNewExec(backend, store, counterFn)
fmt.Printf("Step 1: %v\n", exec.MustCall1())
fmt.Printf("Step 2: %v\n", exec.MustCall1())
fmt.Printf("Step 3: %v\n", exec.MustCall1())
```
<div align="right"><small><a href="https://github.com/gomlx/gomlx/blob/main/examples/gomlx.github.io/core-concepts/store/main.go#L39">(See source)</a></small></div>

Output:

<!-- sync_code: file=core_concepts/store/main.go output_tag=counter -->
```
Step 1: int32(1)
Step 2: int32(2)
Step 3: int32(3)
```

Inside the model function, `ctx.VariableWithValue` retrieves or initializes the variable, `counterVar.NodeValue(g)` returns the node representing its current value, and `counterVar.SetNodeValue(counter)` updates its value in the store with the computation result.

### Scopes and Hierarchical Parameters

A [Scope](file:///home/janpf/Projects/gomlx/gomlx/ml/model/scope.go) represents a path in the hierarchical store (similar to directories). When building complex model architectures, scopes allow you to separate variables of different layers.

Here is an example of defining a custom `denseLayer` function and applying it to different sub-scopes using `ctx.In`:

<!-- sync_code: file=core_concepts/store/main.go tag=scopes -->
```go
func denseLayer(ctx *model.Scope, x *Node, outputDims int) *Node {
g := x.Graph()
dtype := x.DType()
inputDims := x.Shape().Dimensions[1] // x shape is [batch, inputDims]

// Create weights and biases in the current scope
weights := ctx.VariableWithShape("weights", shapes.Make(dtype, inputDims, outputDims)).NodeValue(g)
biases := ctx.VariableWithShape("biases", shapes.Make(dtype, 1, outputDims)).NodeValue(g)

// Compute x * weights + biases
return Add(Dot(x, weights).Product(), biases)
}

modelFn := func(ctx *model.Scope, x *Node) *Node {
	// Use ctx.In to partition variable names under sub-scopes:
	h := denseLayer(ctx.In("layer1"), x, 3) // variables: /layer1/weights, /layer1/biases
	y := denseLayer(ctx.In("layer2"), h, 1) // variables: /layer2/weights, /layer2/biases
	return y
}
```
<div align="right"><small><a href="https://github.com/gomlx/gomlx/blob/main/examples/gomlx.github.io/core-concepts/store/main.go#L17">(See source)</a></small></div>

Using `ctx.In("layer1")` ensures that the weights and biases of the first layer are stored under paths like `/layer1/weights` and `/layer1/biases`, avoiding conflicts with other layers.

If we run the model function, we can inspect all of the variables registered in the store:

<!-- sync_code: file=core_concepts/store/main.go tag=print_vars -->
```go
// We can inspect all variables in the store:
for v := range store.IterVariables() {
	fmt.Printf("Variable: %s, shape: %s\n", v.Path(), v.Shape())
}
```
<div align="right"><small><a href="https://github.com/gomlx/gomlx/blob/main/examples/gomlx.github.io/core-concepts/store/main.go#L76">(See source)</a></small></div>

Output:

<!-- sync_code: file=core_concepts/store/main.go output_tag=print_vars -->
```
Variable: /counter, shape: (Int32)
Variable: /layer1/weights, shape: (Float32)[2, 3]
Variable: /layer1/biases, shape: (Float32)[1, 3]
Variable: /layer2/weights, shape: (Float32)[3, 1]
Variable: /layer2/biases, shape: (Float32)[1, 1]
Variable: /#rngState, shape: (Uint64)[3]
```

---

## Training a Model

Here is the minimal skeleton of a trainable GoMLX program:

```go
func main() {
    // 1. Connect to hardware
    backend := compute.New()

    // 2. Create a store to hold weights
    store := model.NewStore()

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

Other related topics: **datasets**, **Hyperparameters**, **Losses**, **Optimizers**, **Metrics**. <!-- link to future topics with their own pages -->
