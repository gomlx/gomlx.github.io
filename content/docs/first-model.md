---
title: "Your First Model: MNIST Digit Classifier with GoMLX"
lead: "Build, train, and evaluate a CNN in Go to classify MNIST digits, end to end."
weight: 1
---

Welcome to GoMLX! In this end-to-end tutorial, you will set up your environment, build, train, and evaluate a Convolutional Neural Network (CNN) in Go to classify handwritten digits from the [**MNIST dataset**](https://en.wikipedia.org/wiki/MNIST_database).

This guide bridges high-level machine learning principles with practical, idiomatic Go code, using GoMLX's real, current APIs — every code sample below is synced directly from, and tested as part of, [`examples/gomlx.github.io/mnist/first-model/main.go`](https://github.com/gomlx/gomlx/blob/main/examples/gomlx.github.io/mnist/first-model/main.go) in the `gomlx/gomlx` repository, so it can't drift out of sync with the library.

---

## 1. Prerequisites & Environment Setup

### Step 1: Verify Go Installation

GoMLX currently targets **Go 1.26 or later** (check the `go` directive in [go.mod](https://github.com/gomlx/gomlx/blob/main/go.mod) if you're unsure which version a given release requires).

```bash
go version
```

If Go is not installed, download it from the official [Go Downloads page](https://go.dev/dl/).

{{< callout type="note" >}}
GoMLX's default backend talks to the XLA/PJRT plugin through cgo bindings, so you need `CGO_ENABLED=1` (the default) and a C compiler (`gcc` or `clang`) available on your `PATH`. You don't need to install XLA yourself — see the note on auto-installation below.
{{< /callout >}}

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

`gomlx` is the machine learning library (layers, training loop, datasets). `compute` is the lower-level package — [github.com/gomlx/compute](https://pkg.go.dev/github.com/gomlx/compute) — that owns the `Backend`, `dtypes`, and `shapes` types — you'll import both directly in `main.go`, so both need to be in `go.mod`. (Running `go mod tidy` after you write the code below would also pull in `compute` automatically, since it's a transitive dependency — but adding it explicitly up front is one less thing to think about.)

**On first run**, GoMLX auto-downloads and installs the correct XLA PJRT plugin for your OS/architecture (CPU by default; GPU/TPU if configured) — no manual XLA install needed. If you want to disable that behavior (e.g. for a locked-down CI environment), set the environment variable `GOMLX_NO_AUTO_INSTALL=1`.

---

## 2. Core GoMLX Concepts

GoMLX organizes machine learning workflows around five building blocks:

* **Backend** ([`compute.Backend`](https://pkg.go.dev/github.com/gomlx/compute#Backend)): Executes the actual numerical computation. GoMLX compiles execution graphs via **OpenXLA**, supporting CPU, GPU, and TPU. Created with `compute.MustNew()`.
* **Store & Scope** ([`*model.Store`](https://github.com/gomlx/gomlx/blob/main/ml/model/store.go) / [`*model.Scope`](https://github.com/gomlx/gomlx/blob/main/ml/model/scope.go)): The `Store` owns every trainable variable and hyperparameter in your model, organized hierarchically. A `Scope` is a named "directory" inside that tree (e.g. `conv1/weights`) — you get one from `store.RootScope()` and navigate it with `scope.In("name")`. See [Variables, Hyperparameters & Checkpointing](/docs/variables-and-checkpoints) for the full picture.
* **Graph** (`*graph.Graph` / `*graph.Node`): Symbolic computation nodes, built once and compiled to XLA before any data flows through them. See [Computation Graph](/docs/computation-graph).
* **Model function**: A plain Go function with the signature `func(scope *model.Scope, spec any, inputs []*graph.Node) []*graph.Node`, describing how inputs become predictions.
* **Trainer & Loop** (`*train.Trainer` / `*train.Loop`): `Trainer` wires together the model function, loss, optimizer, and metrics into a single training/eval step; `Loop` repeatedly calls that step over a dataset (by number of steps or by epochs). See [Training Loop](/docs/training-loop).

{{< callout type="note" >}}
If you've seen older GoMLX examples using `context.Context`, note that the library has since moved to the `model.Store`/`model.Scope` pair described above — there is no `ml/context` package in the current API.
{{< /callout >}}

---

## 3. Package Imports

<!-- sync_code: file=mnist/first-model/main.go tag=imports -->
```go
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
```
<div align="right"><small><a href="https://github.com/gomlx/gomlx/blob/main/examples/gomlx.github.io/mnist/first-model/main.go">(See source)</a></small></div>

A few things worth calling out:

* [`github.com/gomlx/gomlx/examples/mnist`](https://github.com/gomlx/gomlx/blob/main/examples/mnist/dataset.go) is GoMLX's own MNIST downloader/loader — it's a real importable package, not sample-only code, so we reuse it instead of writing IDX-file parsing from scratch.
* `. "github.com/gomlx/gomlx/core/graph"` is a **dot import**: it's the idiomatic style used throughout GoMLX's own examples so you can write `Reshape(...)` instead of `graph.Reshape(...)`. It's optional — a regular `"github.com/gomlx/gomlx/core/graph"` import works too, you'll just prefix everything with `graph.`.
* `activation.Relu` lives in its own package, `ml/layers/activation`, separate from both `core/graph` and `ml/layers` — it's easy to assume it's a plain graph op (it doesn't own variables), but GoMLX groups all activation functions there instead.
* The blank import `_ "github.com/gomlx/gomlx/backends/default"` registers the CPU/GPU (XLA) and pure-Go backends so `compute.MustNew()` has something to find. Forgetting this import is a common "no backend available" error.

---

## 4. Defining the Neural Network Architecture

We build a small **Convolutional Neural Network (CNN)**:

1. Two **Conv2D → ReLU → MaxPool** feature-extraction blocks.
2. A **flatten** step (a reshape) to turn the 2D feature maps into a 1D vector per example.
3. A **Dense** hidden layer with ReLU.
4. An output **Dense** layer producing raw logits for the 10 digit classes (0–9).

<!-- sync_code: file=mnist/first-model/main.go tag=model -->
```go
// ConvModel defines a simple CNN architecture for MNIST digit classification.
// This is a "model function": GoMLX calls it once, while building the computation graph,
// not once per example -- the returned Nodes are symbolic, not actual numbers yet.
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
<div align="right"><small><a href="https://github.com/gomlx/gomlx/blob/main/examples/gomlx.github.io/mnist/first-model/main.go#L72">(See source)</a></small></div>

{{< callout type="note" >}}
**Parameter scoping (`scope.In`):** `scope.In("conv1")` creates a nested namespace for variables created inside it (e.g. `conv1/weights`, `conv1/biases`), preventing name collisions between layers that would otherwise both try to create a variable called `weights`.

**`MaxPool` comes from `core/graph`, `Relu` from `ml/layers/activation`:** pooling and reshaping are plain graph operations (they don't own trainable variables), so they live alongside `Reshape` in the `graph` package. Activations get their own package, `ml/layers/activation`. Only operations that own variables — `Convolution`, `Dense` — live in `ml/layers` proper and take a `*model.Scope` as their first argument.
{{< /callout >}}

---

## 5. Preparing & Loading the Dataset

[`examples/mnist`](https://github.com/gomlx/gomlx/blob/main/examples/mnist/dataset.go) handles downloading and decoding the IDX files for you. We still need to (a) pick a cache directory, (b) trigger the download, and (c) turn the raw, unbatched dataset it returns into a batched, shuffled training stream and a batched evaluation stream.

<!-- sync_code: file=mnist/first-model/main.go tag=dataset -->
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
<div align="right"><small><a href="https://github.com/gomlx/gomlx/blob/main/examples/gomlx.github.io/mnist/first-model/main.go#L35">(See source)</a></small></div>

{{< callout type="note" >}}
**Why not `Infinite(true)`?** You'll see `.Infinite(true)` in some GoMLX examples that drive the loop with `loop.RunSteps(ds, numSteps)` (a fixed number of steps, dataset repeats forever). We're using `loop.RunEpochs(ds, epochs)` instead, which iterates the dataset to exhaustion once per epoch — so the dataset must be finite.
{{< /callout >}}

---

## 6. Training & Evaluation Pipeline

[`train.Trainer`](https://github.com/gomlx/gomlx/blob/main/ml/train/trainer.go) wires the model function, loss, optimizer, and metrics together into a single step function. [`train.Loop`](https://github.com/gomlx/gomlx/blob/main/ml/train/loop.go) drives that step function over a dataset.

<!-- sync_code: file=mnist/first-model/main.go tag=training -->
```go
backend := compute.MustNew()
defer backend.Finalize()
fmt.Printf("Backend: %s (%s)\n", backend.Name(), backend.Description())

store := model.NewStore()
trainDS, testDS := prepareDatasets(backend, dataDir)

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

const epochs = 5
loop := train.NewLoop(trainer)
// AttachProgressBar gives you a live progress bar with loss/metric values -- there's no
// need to print them by hand.
commandline.AttachProgressBar(loop)

fmt.Printf("Starting training for %d epochs...\n", epochs)
if _, err := loop.RunEpochs(trainDS, epochs); err != nil {
	log.Fatalf("Training loop failed: %+v", err)
}

fmt.Println("\nEvaluating model performance on test set...")
if err := commandline.ReportEval(trainer, testDS); err != nil {
	log.Fatalf("Evaluation failed: %+v", err)
}
```
<div align="right"><small><a href="https://github.com/gomlx/gomlx/blob/main/examples/gomlx.github.io/mnist/first-model/main.go#L107">(See source)</a></small></div>

### Source references for this section

[`loss.SparseCategoricalCrossEntropyLogits`](https://github.com/gomlx/gomlx/blob/main/ml/train/loss/loss.go), [`metric.NewSparseCategoricalAccuracy`](https://github.com/gomlx/gomlx/blob/main/ml/train/metric/metric.go), [`optimizer.Adam`](https://github.com/gomlx/gomlx/blob/main/ml/train/optimizer/adam.go), [`Loop.RunEpochs`](https://github.com/gomlx/gomlx/blob/main/ml/train/loop.go), [`Trainer.Eval`](https://github.com/gomlx/gomlx/blob/main/ml/train/trainer.go) (called internally by `commandline.ReportEval`), [`commandline.AttachProgressBar`](https://github.com/gomlx/gomlx/blob/main/ui/commandline/progressbar.go).

---

## 7. Running Your Code & Output

Clone the repo and run the tutorial's source directly:

```bash
go run github.com/gomlx/gomlx/examples/gomlx.github.io/mnist/first-model@latest
```

Or copy the snippets from sections 3–6 above into a single `main.go` in your own module and run `go run main.go`.

The first run will download the MNIST files (~11MB) to `~/mnist_data` and auto-install the XLA PJRT plugin if it isn't already cached, so it'll be slower than subsequent runs. `commandline.AttachProgressBar` gives you a live-updating table plus a progress bar during training, followed by the evaluation report. This is real, complete output from an actual 5-epoch run on CPU:

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

{{< callout type="note" >}}
Treat exact numbers as illustrative, not a guarantee — they depend on random weight initialization, batch order, and hardware, and will vary a bit (typically within a percent or so of accuracy) between runs. This run took a bit under 5 minutes on CPU for 2,340 steps (5 epochs × 468 steps/epoch).
{{< /callout >}}

---

## 8. Checkpointing the Model

Training runs end when your program exits, and so far, so does the model — nothing is saved to disk. Two lines added to the imports and a few more around the `Loop` are all it takes to persist progress and resume it later.

### Add the checkpoint import

<!-- sync_code: file=mnist/first-model/main.go tag=checkpoint_imports -->
```go
"time"
"github.com/gomlx/gomlx/ml/model/checkpoint"
```
<div align="right"><small><a href="https://github.com/gomlx/gomlx/blob/main/examples/gomlx.github.io/mnist/first-model/main.go">(See source)</a></small></div>

### Save periodically during training

Insert this right after creating the `Loop` (before `commandline.AttachProgressBar`):

<!-- sync_code: file=mnist/first-model/main.go tag=checkpoint_setup -->
```go
// Save a checkpoint every minute, keeping the 3 most recent ones. checkpoint.Build
// also *loads* an existing checkpoint from checkpointDir if one is already there, so
// re-running this program resumes training instead of starting over.
checkpointHandler, err := checkpoint.Build(store).
	DirFromBase("checkpoint", dataDir).
	Keep(3).
	Done()
if err != nil {
	log.Fatalf("Failed to create checkpoint handler: %+v", err)
}
train.PeriodicCallback(loop, time.Minute, true, "saving checkpoint", 100, checkpointHandler.SaveOnStepFn)
```
<div align="right"><small><a href="https://github.com/gomlx/gomlx/blob/main/examples/gomlx.github.io/mnist/first-model/main.go#L133">(See source)</a></small></div>

`checkpoint.Build(store)` does double duty: it *loads* an existing checkpoint from `checkpointDir` if one is already there (so re-running this program resumes training instead of starting from scratch), and prepares the handler used to *save* new ones. `train.PeriodicCallback` then triggers `checkpointHandler.SaveOnStepFn` on a timer — here, once a minute, plus immediately on startup (the `true` argument) — and `Keep(3)` prunes older checkpoints so disk usage doesn't grow unbounded.

{{< callout type="note" >}}
This is intentionally the minimum needed to add checkpointing to this tutorial. For the full picture — `TempDir`, `FromEmbed`, `TakeMean`, compression, lazy loading, and the `gomlx_checkpoints` inspection tool — see [Variables, Hyperparameters & Checkpointing](/docs/variables-and-checkpoints).
{{< /callout >}}

---

## 9. Next Steps & Advanced Features

* **Interactive plotting:** Under [GoNB](https://github.com/janpfeifer/gonb) (a Go kernel for Jupyter), `github.com/gomlx/gomlx/ui/gonb/plotly` gives you live, inline training-curve plots — see [`examples/mnist/train.go`](https://github.com/gomlx/gomlx/blob/main/examples/mnist/train.go) for a real usage example (`plotly.New().WithCheckpoint(...).Dynamic()...`).
* **Hyperparameter management:** `model.Store` supports `Store.SetParam`/`SetParams` and reading params back with `model.GetParamOr(scope, name, default)`, plus CLI flag wiring via `github.com/gomlx/gomlx/ui/commandline` — see how [`examples/mnist/train.go`](https://github.com/gomlx/gomlx/blob/main/examples/mnist/train.go)'s `CreateStore()` predefines tunables like `learning_rate` and `cnn_dropout_rate`, settable from the command line with `-set="learning_rate=0.0005"`. See also [Hyperparameters](/docs/hyperparameters).
* **Regularization:** Add dropout with `layers.DropoutStatic(scope, x, 0.2)` (a fixed rate) or `layers.Dropout(scope, x, dropoutRateNode)` (a rate you can vary, e.g. on/off between train and eval).
* **Inference from a checkpoint:** [`examples/mnist/classifier/classifier.go`](https://github.com/gomlx/gomlx/blob/main/examples/mnist/classifier/classifier.go) is a complete example of loading a checkpointed MNIST model and running inference on a single image.
