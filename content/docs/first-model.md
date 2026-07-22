---
title: "Your First Model: MNIST Digit Classifier with GoMLX"
lead: "Build, train, and evaluate a CNN in Go to classify MNIST digits, end to end."
weight: 1
---

Welcome to GoMLX! In this end-to-end tutorial, you will set up your environment, build, train, and evaluate a Convolutional Neural Network (CNN) in Go to classify handwritten digits from the [**MNIST dataset**](https://en.wikipedia.org/wiki/MNIST_database).

This guide bridges high-level machine learning principles with practical, idiomatic Go code, using GoMLX's real, current APIs — every code sample here is checked against, and runnable as, [`examples/mnist/quickstart/main.go`](https://github.com/gomlx/gomlx/blob/main/examples/mnist/quickstart/main.go) in this repository.

---

## 1. Prerequisites & Environment Setup

### Step 1: Verify Go Installation

GoMLX currently targets **Go 1.26 or later** (check the `go` directive in [go.mod](https://github.com/gomlx/gomlx/blob/main/go.mod) if you're unsure which version a given release requires).

```bash
go version
```

If Go is not installed, download it from the official [Go Downloads page](https://go.dev/dl/).

> **Note on CGO:** GoMLX's default backend talks to the XLA/PJRT plugin through cgo bindings, so you need `CGO_ENABLED=1` (the default) and a C compiler (`gcc` or `clang`) available on your `PATH`. You don't need to install XLA yourself — see the note on auto-installation below.

### Step 2: Initialize Your Go Project

```bash
mkdir gomlx-mnist-quickstart
cd gomlx-mnist-quickstart
go mod init gomlx-mnist-quickstart
```

### Step 3: Fetch GoMLX Dependencies

```bash
go get github.com/gomlx/gomlx@latest
go get github.com/gomlx/compute@latest
go get github.com/gomlx/gomlx/examples/mnist@latest
```

> ⚠️ **Publishing blocker — do not ship this tutorial until this is resolved.** [`examples/`](https://github.com/gomlx/gomlx/blob/main/examples/go.mod) is a separate Go module from the root `gomlx` module, tagged independently. Its latest published tag, `examples/v0.27.3` (2026-04-16), predates a refactor that landed in root module `v0.28.0` (2026-07-21) — renaming `ml/context` → `ml/model`, `graph` → `core/graph`, and singularizing `ml/train/losses|metrics|optimizers` → `loss|metric|optimizer`. Verified by actually running it: `go get` of both `@latest` together resolves `examples` to the stale tag, and the build fails with ~20 "does not contain package" errors — no `@main`/commit-pin workaround reliably fixes it either. **Ask the maintainers to cut a new `examples` tag (e.g. `examples/v0.28.0`) against current `main` before this page goes live.** Until then, the only way to run this code is inside a local checkout of this repo (see [`examples/mnist/quickstart/`](https://github.com/gomlx/gomlx/blob/main/examples/mnist/quickstart/main.go), which builds today because the `examples` module's own internal dependency pin is already fresh) or via a `replace` directive to a local clone.

`gomlx` is the machine learning library (layers, training loop, datasets). `compute` is the lower-level package — [github.com/gomlx/compute](https://pkg.go.dev/github.com/gomlx/compute) — that owns the `Backend`, `dtypes`, and `shapes` types — you'll import both directly in `main.go`, so both need to be in `go.mod`. (Running `go mod tidy` after you write the code below would also pull in `compute` automatically, since it's a transitive dependency — but adding it explicitly up front is one less thing to think about.)

**On first run**, GoMLX auto-downloads and installs the correct XLA PJRT plugin for your OS/architecture (CPU by default; GPU/TPU if configured) — no manual XLA install needed. If you want to disable that behavior (e.g. for a locked-down CI environment), set the environment variable `GOMLX_NO_AUTO_INSTALL=1`.

---

## 2. Core GoMLX Concepts

GoMLX organizes machine learning workflows around five building blocks:

* **Backend** ([`compute.Backend`](https://pkg.go.dev/github.com/gomlx/compute#Backend)): Executes the actual numerical computation. GoMLX compiles execution graphs via **OpenXLA**, supporting CPU, GPU, and TPU. Created with `compute.MustNew()`.
* **Store & Scope** ([`*model.Store`](https://github.com/gomlx/gomlx/blob/main/ml/model/store.go#L67) / [`*model.Scope`](https://github.com/gomlx/gomlx/blob/main/ml/model/scope.go#L101)): The `Store` owns every trainable variable and hyperparameter in your model, organized hierarchically. A `Scope` is a named "directory" inside that tree (e.g. `conv1/weights`) — you get one from `store.RootScope()` and navigate it with `scope.In("name")`.
* **Graph** ([`*graph.Graph` / `*graph.Node`](https://github.com/gomlx/gomlx/tree/main/core/graph)): Symbolic computation nodes, built once and compiled to XLA before any data flows through them.
* **Model function**: A plain Go function with the signature `func(scope *model.Scope, spec any, inputs []*graph.Node) []*graph.Node`, describing how inputs become predictions.
* **Trainer & Loop** ([`*train.Trainer`](https://github.com/gomlx/gomlx/blob/main/ml/train/trainer.go#L180) / [`*train.Loop`](https://github.com/gomlx/gomlx/blob/main/ml/train/loop.go#L92)): `Trainer` wires together the model function, loss, optimizer, and metrics into a single training/eval step; `Loop` repeatedly calls that step over a dataset (by number of steps or by epochs).

> If you've seen older GoMLX examples using `context.Context`, note that the library has since moved to the `model.Store`/`model.Scope` pair described above — there is no `ml/context` package in the current API.

---

## 3. Package Imports

```go
package main

import (
	"fmt"
	"log"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/examples/mnist"
	"github.com/gomlx/gomlx/ml/dataset"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/activation"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/loss"
	"github.com/gomlx/gomlx/ml/train/metric"
	"github.com/gomlx/gomlx/ml/train/optimizer"
	"github.com/gomlx/gomlx/support/fsutil"
	"github.com/gomlx/gomlx/ui/commandline"

	// Registers the default backends (XLA and the pure-Go fallback) with compute.MustNew().
	_ "github.com/gomlx/gomlx/backends/default"
)
```

A few things worth calling out:

* [`github.com/gomlx/gomlx/examples/mnist`](https://github.com/gomlx/gomlx/blob/main/examples/mnist/dataset.go) is GoMLX's own MNIST downloader/loader — it's a real importable package, not sample-only code, so we reuse it instead of writing IDX-file parsing from scratch.
* `. "github.com/gomlx/gomlx/core/graph"` is a **dot import**: it's the idiomatic style used throughout GoMLX's own examples (see [`examples/linear/linear.go`](https://github.com/gomlx/gomlx/blob/main/examples/linear/linear.go)) so you can write `Reshape(...)` instead of `graph.Reshape(...)`. It's optional — a regular `"github.com/gomlx/gomlx/core/graph"` import works too, you'll just prefix everything with `graph.`.
* [`activation.Relu`](https://github.com/gomlx/gomlx/blob/main/ml/layers/activation/activation.go#L161) lives in its own package, `ml/layers/activation`, separate from both `core/graph` and `ml/layers` — it's easy to assume it's a plain graph op (it doesn't own variables), but GoMLX groups all activation functions there instead.
* The blank import `_ "github.com/gomlx/gomlx/backends/default"` registers the CPU/GPU (XLA) and pure-Go backends so `compute.MustNew()` has something to find. Forgetting this import is a common "no backend available" error.

---

## 4. Defining the Neural Network Architecture

We build a small **Convolutional Neural Network (CNN)**:

1. Two **Conv2D → ReLU → MaxPool** feature-extraction blocks.
2. A **flatten** step (a reshape) to turn the 2D feature maps into a 1D vector per example.
3. A **Dense** hidden layer with ReLU.
4. An output **Dense** layer producing raw logits for the 10 digit classes (0–9).

```go
// ConvModel defines a simple CNN architecture for MNIST digit classification.
// This is a "model function": GoMLX calls it once, while building the computation graph,
// not once per example — the returned Nodes are symbolic, not actual numbers yet.
func ConvModel(scope *model.Scope, spec any, inputs []*Node) []*Node {
	// inputs[0] shape: [BatchSize, 28, 28, 1] (grayscale).
	images := inputs[0]
	batchSize := images.Shape().Dimensions[0]

	// Block 1: Conv2D (16 filters, 3x3 kernel, "same" padding keeps 28x28) -> ReLU -> MaxPool 2x2 -> 14x14.
	x := layers.Convolution(scope.In("conv1"), images).Filters(16).KernelSize(3).PadSame().Done()
	x = activation.Relu(x)
	x = MaxPool(x).Window(2).Done()

	// Block 2: Conv2D (32 filters) -> ReLU -> MaxPool 2x2 -> 7x7.
	x = layers.Convolution(scope.In("conv2"), x).Filters(32).KernelSize(3).PadSame().Done()
	x = activation.Relu(x)
	x = MaxPool(x).Window(2).Done()

	// Flatten: [BatchSize, 7, 7, 32] -> [BatchSize, 7*7*32]. GoMLX doesn't have a separate
	// Flatten layer -- a Reshape with -1 for the last dimension does the job.
	x = Reshape(x, batchSize, -1)

	// Fully-connected hidden layer (128 units) + ReLU.
	x = layers.Dense(scope.In("dense1"), x, true, 128)
	x = activation.Relu(x)

	// Output layer: logits for the 10 classes. No activation here -- the loss function
	// (SparseCategoricalCrossEntropyLogits) applies softmax internally, on the logits, which is
	// more numerically stable than applying softmax yourself and feeding probabilities to the loss.
	logits := layers.Dense(scope.In("logits"), x, true, mnist.NumClasses)

	return []*Node{logits}
}
```

> **Parameter scoping (`scope.In`):** [`scope.In("conv1")`](https://github.com/gomlx/gomlx/blob/main/ml/model/scope.go#L101) creates a nested namespace for variables created inside it (e.g. `conv1/weights`, `conv1/biases`), preventing name collisions between layers that would otherwise both try to create a variable called `weights`.

> **`MaxPool` comes from `core/graph`, `Relu` from `ml/layers/activation`:** pooling and reshaping are plain graph operations (they don't own trainable variables), so they live alongside [`Reshape`](https://github.com/gomlx/gomlx/blob/main/core/graph/ops.go#L757) in the [`graph`](https://github.com/gomlx/gomlx/blob/main/core/graph/ops_pool.go#L56) package. Activations get their own package, `ml/layers/activation`. Only operations that own variables — [`Convolution`](https://github.com/gomlx/gomlx/blob/main/ml/layers/convolution.go#L56), [`Dense`](https://github.com/gomlx/gomlx/blob/main/ml/layers/layers.go#L71) — live in `ml/layers` proper and take a `*model.Scope` as their first argument.

---

## 5. Preparing & Loading the Dataset

[`examples/mnist`](https://github.com/gomlx/gomlx/blob/main/examples/mnist/dataset.go) handles downloading and decoding the IDX files for you. We still need to (a) pick a cache directory, (b) trigger the download, and (c) turn the raw, unbatched dataset it returns into a batched, shuffled training stream and a batched evaluation stream.

```go
const batchSize = 128

// prepareDatasets ensures MNIST is downloaded, then returns batched train/test datasets.
func prepareDatasets(backend compute.Backend, dataDir string) (trainDS, testDS train.Dataset) {
	dataDir = fsutil.MustReplaceTildeInDir(dataDir)

	// mnist.Download is idempotent: it verifies checksums and skips files already on disk.
	if err := mnist.Download(dataDir); err != nil {
		log.Fatalf("Failed to download MNIST dataset: %+v", err)
	}

	// mnist.NewDataset returns the *whole* split as a single in-memory, unbatched dataset --
	// we still need to batch (and, for training, shuffle) it ourselves.
	rawTrain, err := mnist.NewDataset(backend, "MNIST Train", dataDir, "train", dtypes.Float32)
	if err != nil {
		log.Fatalf("Failed to load training dataset: %+v", err)
	}
	rawTest, err := mnist.NewDataset(backend, "MNIST Test", dataDir, "test", dtypes.Float32)
	if err != nil {
		log.Fatalf("Failed to load test dataset: %+v", err)
	}

	// dropIncompleteBatch=true for training keeps every batch a fixed size, which XLA prefers
	// (it recompiles the graph whenever it sees a new shape). Shuffle() reorders examples each epoch.
	trainDS = rawTrain.Shuffle().BatchSize(batchSize, true)
	// For evaluation we want to see every example, so we don't drop the last, short batch.
	testDS = rawTest.BatchSize(batchSize, false)
	return
}
```

> **Why not `Infinite(true)`?** You'll see `.Infinite(true)` in some GoMLX examples that drive the loop with `loop.RunSteps(ds, numSteps)` (a fixed number of steps, dataset repeats forever). We're using `loop.RunEpochs(ds, epochs)` instead, which iterates the dataset to exhaustion once per epoch — so the dataset must be finite. Source: [`mnist.Download`](https://github.com/gomlx/gomlx/blob/main/examples/mnist/dataset.go#L127), [`mnist.NewDataset`](https://github.com/gomlx/gomlx/blob/main/examples/mnist/dataset.go#L154).

---

## 6. Training & Evaluation Pipeline

[`train.Trainer`](https://github.com/gomlx/gomlx/blob/main/ml/train/trainer.go#L180) wires the model function, loss, optimizer, and metrics together into a single step function. [`train.Loop`](https://github.com/gomlx/gomlx/blob/main/ml/train/loop.go#L92) drives that step function over a dataset.

```go
func main() {
	// 1. Create the backend (XLA if available, otherwise the pure-Go fallback).
	backend := compute.MustNew()
	defer backend.Finalize()
	fmt.Printf("Backend: %s (%s)\n", backend.Name(), backend.Description())

	// 2. Create the Store that will hold every trainable variable in the model.
	store := model.NewStore()

	// 3. Load and prepare the MNIST data streams.
	trainDS, testDS := prepareDatasets(backend, "~/mnist_data")

	// 4. Build the Trainer: it ties together the model function, loss, optimizer and metrics.
	accuracyMetric := metric.NewSparseCategoricalAccuracy("Accuracy", "acc")
	trainer := train.NewTrainer(
		backend,
		store,
		ConvModel,
		loss.SparseCategoricalCrossEntropyLogits, // cross-entropy loss for integer labels
		optimizer.Adam().LearningRate(1e-3).Done(),
		[]metric.Interface{accuracyMetric}, // metrics reported during training
		[]metric.Interface{accuracyMetric}, // metrics reported during evaluation
	)

	// 5. Run training for a fixed number of epochs. AttachProgressBar gives you a live
	// progress bar with loss/metric values -- there's no need to print them by hand.
	const epochs = 5
	loop := train.NewLoop(trainer)
	commandline.AttachProgressBar(loop)

	fmt.Printf("Starting training for %d epochs...\n", epochs)
	if _, err := loop.RunEpochs(trainDS, epochs); err != nil {
		log.Fatalf("Training loop failed: %+v", err)
	}

	// 6. Evaluate on the held-out test set.
	fmt.Println("\nEvaluating model performance on test set...")
	if err := commandline.ReportEval(trainer, testDS); err != nil {
		log.Fatalf("Evaluation failed: %+v", err)
	}
}
```

Source references for this section: [`loss.SparseCategoricalCrossEntropyLogits`](https://github.com/gomlx/gomlx/blob/main/ml/train/loss/loss.go#L373), [`metric.NewSparseCategoricalAccuracy`](https://github.com/gomlx/gomlx/blob/main/ml/train/metric/metric.go#L513), [`optimizer.Adam`](https://github.com/gomlx/gomlx/blob/main/ml/train/optimizer/adam.go#L65), [`Loop.RunEpochs`](https://github.com/gomlx/gomlx/blob/main/ml/train/loop.go#L437), [`Trainer.Eval`](https://github.com/gomlx/gomlx/blob/main/ml/train/trainer.go#L874) (called internally by [`commandline.ReportEval`](https://github.com/gomlx/gomlx/blob/main/ui/commandline/commandline.go#L15)), [`commandline.AttachProgressBar`](https://github.com/gomlx/gomlx/blob/main/ui/commandline/progressbar.go#L183).

---

## 7. Complete Runnable `main.go`

The full file lives in this repository at [`examples/mnist/quickstart/main.go`](https://github.com/gomlx/gomlx/blob/main/examples/mnist/quickstart/main.go) — clone the repo and run `go run ./examples/mnist/quickstart` directly, or copy the code below into your own module:

```go
package main

import (
	"fmt"
	"log"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/examples/mnist"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/activation"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/loss"
	"github.com/gomlx/gomlx/ml/train/metric"
	"github.com/gomlx/gomlx/ml/train/optimizer"
	"github.com/gomlx/gomlx/support/fsutil"
	"github.com/gomlx/gomlx/ui/commandline"

	_ "github.com/gomlx/gomlx/backends/default"
)

const batchSize = 128

// ConvModel defines a simple CNN architecture for MNIST digit classification.
func ConvModel(scope *model.Scope, spec any, inputs []*Node) []*Node {
	images := inputs[0]
	batchSize := images.Shape().Dimensions[0]

	x := layers.Convolution(scope.In("conv1"), images).Filters(16).KernelSize(3).PadSame().Done()
	x = activation.Relu(x)
	x = MaxPool(x).Window(2).Done()

	x = layers.Convolution(scope.In("conv2"), x).Filters(32).KernelSize(3).PadSame().Done()
	x = activation.Relu(x)
	x = MaxPool(x).Window(2).Done()

	x = Reshape(x, batchSize, -1)

	x = layers.Dense(scope.In("dense1"), x, true, 128)
	x = activation.Relu(x)

	logits := layers.Dense(scope.In("logits"), x, true, mnist.NumClasses)
	return []*Node{logits}
}

// prepareDatasets ensures MNIST is downloaded, then returns batched train/test datasets.
func prepareDatasets(backend compute.Backend, dataDir string) (trainDS, testDS train.Dataset) {
	dataDir = fsutil.MustReplaceTildeInDir(dataDir)

	if err := mnist.Download(dataDir); err != nil {
		log.Fatalf("Failed to download MNIST dataset: %+v", err)
	}

	rawTrain, err := mnist.NewDataset(backend, "MNIST Train", dataDir, "train", dtypes.Float32)
	if err != nil {
		log.Fatalf("Failed to load training dataset: %+v", err)
	}
	rawTest, err := mnist.NewDataset(backend, "MNIST Test", dataDir, "test", dtypes.Float32)
	if err != nil {
		log.Fatalf("Failed to load test dataset: %+v", err)
	}

	trainDS = rawTrain.Shuffle().BatchSize(batchSize, true)
	testDS = rawTest.BatchSize(batchSize, false)
	return
}

func main() {
	backend := compute.MustNew()
	defer backend.Finalize()
	fmt.Printf("Backend: %s (%s)\n", backend.Name(), backend.Description())

	store := model.NewStore()
	trainDS, testDS := prepareDatasets(backend, "~/mnist_data")

	accuracyMetric := metric.NewSparseCategoricalAccuracy("Accuracy", "acc")
	trainer := train.NewTrainer(
		backend,
		store,
		ConvModel,
		loss.SparseCategoricalCrossEntropyLogits,
		optimizer.Adam().LearningRate(1e-3).Done(),
		[]metric.Interface{accuracyMetric},
		[]metric.Interface{accuracyMetric},
	)

	const epochs = 5
	loop := train.NewLoop(trainer)
	commandline.AttachProgressBar(loop)

	fmt.Printf("Starting training for %d epochs...\n", epochs)
	if _, err := loop.RunEpochs(trainDS, epochs); err != nil {
		log.Fatalf("Training loop failed: %+v", err)
	}

	fmt.Println("\nEvaluating model performance on test set...")
	if err := commandline.ReportEval(trainer, testDS); err != nil {
		log.Fatalf("Evaluation failed: %+v", err)
	}
}
```

---

## 8. Running Your Code & Output

```bash
go run main.go
```

The first run will download the MNIST files (~11MB) to `~/mnist_data` and auto-install the XLA PJRT plugin if it isn't already cached, so it'll be slower than subsequent runs. `commandline.AttachProgressBar` gives you a live-updating table plus a progress bar during training, followed by the evaluation report. This is the real, complete output from an actual 5-epoch run of `examples/mnist/quickstart` on CPU:

```text
Backend: xla (xla:cpu - PJRT "cpu" plugin ...)
Starting training for 5 epochs...
        ╭────────────────────────────┬─────────────╮
        │                Global Step │ 999 of 2_340│
        │ Median train step duration │ 106.2ms     │
        │                       Loss │ 0.0405      │
        │        Moving Average Loss │ 0.0478      │
        │                   Accuracy │ 96.35%      │
        ╰────────────────────────────┴─────────────╯
        100% [========================================] (9 steps/s)

Evaluating model performance on test set...
Results on MNIST Test:
	Mean Loss (#loss): 0.0494
	Accuracy (acc): 98.43%
```

> Treat exact numbers as illustrative, not a guarantee — they depend on random weight initialization, batch order, and hardware, and will vary a bit (typically within a percent or so of accuracy) between runs. This run took a bit under 5 minutes on CPU for 2,340 steps (5 epochs × 468 steps/epoch).

---

## 9. Next Steps & Advanced Features

* **Interactive plotting:** Under [GoNB](https://github.com/janpfeifer/gonb) (a Go kernel for Jupyter), `github.com/gomlx/gomlx/ui/gonb/plotly` gives you live, inline training-curve plots — see [`examples/mnist/train.go`](https://github.com/gomlx/gomlx/blob/main/examples/mnist/train.go) for a real usage example (`plotly.New().WithCheckpoint(...).Dynamic()...`).
* **Hyperparameter management:** `model.Store` supports `Store.SetParam`/`SetParams` and reading params back with `model.GetParamOr(scope, name, default)`, plus CLI flag wiring via `github.com/gomlx/gomlx/ui/commandline` — see how [`examples/mnist/train.go`](https://github.com/gomlx/gomlx/blob/main/examples/mnist/train.go)'s `CreateStore()` predefines tunables like `learning_rate` and `cnn_dropout_rate`, settable from the command line with `-set="learning_rate=0.0005"`.
* **Regularization:** Add dropout with `layers.DropoutStatic(scope, x, 0.2)` (a fixed rate) or `layers.Dropout(scope, x, dropoutRateNode)` (a rate you can vary, e.g. on/off between train and eval) — see [`ml/layers/layers.go`](https://github.com/gomlx/gomlx/blob/main/ml/layers/layers.go#L319).
* **Persisting models:** [`github.com/gomlx/gomlx/ml/model/checkpoint`](https://github.com/gomlx/gomlx/blob/main/ml/model/checkpoint/checkpoint.go) — `checkpoint.Build(store).DirFromBase(path, dataDir).Keep(n).Done()` to save periodically, `checkpoint.Load(store).Dir(path).Done()` to restore. [`examples/mnist/classifier/classifier.go`](https://github.com/gomlx/gomlx/blob/main/examples/mnist/classifier/classifier.go) is a complete example of loading a checkpointed MNIST model and running inference on a single image.

---

## Where This Lives in the Codebase

| Concept | Package | Source |
|---|---|---|
| Backend | `compute` | [github.com/gomlx/compute](https://pkg.go.dev/github.com/gomlx/compute) (separate module) |
| Store / Scope | `ml/model` | [`store.go`](https://github.com/gomlx/gomlx/blob/main/ml/model/store.go), [`scope.go`](https://github.com/gomlx/gomlx/blob/main/ml/model/scope.go) |
| Graph ops (Reshape, MaxPool, …) | `core/graph` | [`core/graph/`](https://github.com/gomlx/gomlx/tree/main/core/graph) |
| Convolution, Dense | `ml/layers` | [`convolution.go`](https://github.com/gomlx/gomlx/blob/main/ml/layers/convolution.go), [`layers.go`](https://github.com/gomlx/gomlx/blob/main/ml/layers/layers.go) |
| Activations | `ml/layers/activation` | [`activation.go`](https://github.com/gomlx/gomlx/blob/main/ml/layers/activation/activation.go) |
| Trainer / Loop | `ml/train` | [`trainer.go`](https://github.com/gomlx/gomlx/blob/main/ml/train/trainer.go), [`loop.go`](https://github.com/gomlx/gomlx/blob/main/ml/train/loop.go) |
| Losses | `ml/train/loss` | [`loss.go`](https://github.com/gomlx/gomlx/blob/main/ml/train/loss/loss.go) |
| Metrics | `ml/train/metric` | [`metric.go`](https://github.com/gomlx/gomlx/blob/main/ml/train/metric/metric.go) |
| Optimizers | `ml/train/optimizer` | [`adam.go`](https://github.com/gomlx/gomlx/blob/main/ml/train/optimizer/adam.go) |
| Checkpointing | `ml/model/checkpoint` | [`checkpoint.go`](https://github.com/gomlx/gomlx/blob/main/ml/model/checkpoint/checkpoint.go) |
| MNIST loader | `examples/mnist` | [`dataset.go`](https://github.com/gomlx/gomlx/blob/main/examples/mnist/dataset.go) |
| CLI progress bar / eval report | `ui/commandline` | [`progressbar.go`](https://github.com/gomlx/gomlx/blob/main/ui/commandline/progressbar.go), [`commandline.go`](https://github.com/gomlx/gomlx/blob/main/ui/commandline/commandline.go) |
| This tutorial's runnable source | `examples/mnist/quickstart` | [`main.go`](https://github.com/gomlx/gomlx/blob/main/examples/mnist/quickstart/main.go) |
