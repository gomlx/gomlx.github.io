---
title: "Working with Datasets"
section: "Training & Layers"
weight: 25
---

In GoMLX, the data loading and preprocessing pipeline is modeled around the `train.Dataset` interface. The dataset package provides multiple high-level wrapper datasets to batch, shuffle, transform, and optimize data uploads to hardware accelerators (GPUs/TPUs).

---

## The `Dataset` Interface

A dataset is represented by a simple interface that knows its name and returns a standard Go 1.23 iterator (`iter.Seq2`) yielding batches:

```go
type Dataset interface {
	// Name identifies the dataset for logging, plots, and debugging.
	Name() string

	// Iter returns a stateful iterator over the dataset.
	// Calling Iter() resets the iteration (e.g., at the start of a new epoch).
	Iter() iter.Seq2[Batch, error]
}
```

Each iteration yields a `train.Batch` struct:

```go
type Batch struct {
	// Inputs and Labels are slices of multi-dimensional tensors.
	Inputs, Labels []*tensors.Tensor

	// Spec defines custom task specifications (e.g. for multi-task learning).
	// If the Spec value changes, GoMLX compiles a new computation graph.
	Spec any
}
```

---

## Iterator Invariants & Tensor Ownership

Tensors yielded by a dataset are concrete data structures stored in device memory. Because of this, GoMLX enforces a strict contract regarding tensor life cycles:

### 1. Ownership Transfer
When you pull a `Batch` from the iterator, **ownership of the tensors in `Inputs` and `Labels` is transferred to the caller**. The dataset will not free or reuse those specific tensor memory buffers.

### 2. Mandatory Finalization
Since the Go Garbage Collector cannot track allocations in device memory (like VRAM on a GPU/TPU), you must explicitly free the tensors to prevent memory leaks:
* **Inside `train.Trainer`**: If you pass the dataset to `train.Loop` or `trainer.TrainStep`, the trainer automatically takes over ownership and calls `batch.Finalize()` at the end of the execution step.
* **Outside `train.Trainer`**: If you consume the iterator manually (e.g., inside custom inference or debugging loops), you **must** call `batch.Finalize()` or call `.MustFinalizeAll()` on each individual tensor when you are done with them.

```go
// Manual dataset consumption example
next, stop := iter.Pull2(myDataset.Iter())
defer stop()

for {
	batch, err, ok := next()
	if !ok {
		break
	}
	if err != nil {
		log.Fatalf("failed reading: %v", err)
	}

	// 1. Process batch...
	doSomething(batch.Inputs)

	// 2. Mandatory: free device memory
	batch.Finalize()
}
```

---

## Optimization & Meta-Datasets

The `github.com/gomlx/gomlx/ml/dataset` package provides several "meta-datasets" that wrap an existing dataset to optimize preprocessing and memory transfers.

These allows one with very little effort to easily create, parallelize, buffer and transform a dataset from any data, large or small.

| Meta-Dataset | Creation Helper | Description |
| :--- | :--- | :--- |
| **`InMemory`** | `InMemoryFromData` | Loads the entire dataset into CPU/device memory. It slices batches **directly on the device** using a JIT-compiled gather graph, avoiding slow host-to-device copies during training loops. |
| **`Take`** | `Take` | Wraps a dataset to limit it to only the first `N` batches. Extremely useful for testing code execution on a small subset. |
| **`Buffer`** | `NewBuffer` | Runs the underlying dataset iterator in a background goroutine and buffers batches in a channel, smoothing out variations in disk or network IO. |
| **`OnDevice`** | `NewOnDevice` | Spawns a background worker to upload batches to the accelerator device in parallel with training, hiding PCIe copy latency. |
| **`Map` / `MapOnHost`** | `Map` / `MapOnHost` | Applies a user-defined transformation function to each batch on-the-fly (e.g., for online data augmentation). |
| **`Distributed`** | `NewDistributedAccumulator` | Shards batch data across multiple mesh devices for distributed multi-GPU/TPU training. |

---

## Code Example: UCI-Adult Dataset

The UCI-Adult census example (`github.com/gomlx/gomlx/examples/adult`) dataset is pretty small and can be fully read
into memory. It demonstrates how to construct and configure an `InMemoryDataset` for model training and evaluation.

### 1. Constructing the Dataset
The dataset is loaded as raw categorical and continuous values (`adult.RawData` structure), converted to device tensors,
and wrapped into an `InMemoryDataset` using `dataset.InMemoryFromData`:

```go
func NewDataset(backend compute.Backend, rawData *RawData, name string) *dataset.InMemoryDataset {
	// rawData.CreateTensors returns pre-allocated flat tensors
	tensorData := rawData.CreateTensors(backend)

	// Create InMemoryDataset: inputs are Categorical, Continuous, and Weights; labels are Labels
	ds, err := dataset.InMemoryFromData(backend, name,
		[]any{tensorData.CategoricalTensor, tensorData.ContinuousTensor, tensorData.WeightsTensor},
		[]any{tensorData.LabelsTensor})
	if err != nil {
		panic(errors.WithMessagef(err, "failed to create UCI Adult dataset"))
	}
	return ds
}
```

### 2. Configuring Iteration Modes
Once created, you configure the dataset in different ways for training (shuffle, infinitely looping) and evaluation (no shuffling, one epoch).

Note that `InMemoryDataset` supports `.Copy()` so you can reuse the same in-memory data for training and evaluation with different batch sizes:

```go
// A. Create base in-memory dataset for testing --the default configuration
//    has no shuffling and yields one epoch only.
testEvalDS  := adult.NewDataset(backend, adult.Data.Test, "test").BatchSize(batchSize, false)

// B. Eval on training: no shuffling and one epoch.
baseTrainDS := adult.NewDataset(backend, adult.Data.Train, "batched train")
trainEvalDS := baseTrainDS.BatchSize(batchSize, false)

// C. Copy the underlying data (shallow, only references are copied), and configure
//    to loop indefinitely and shuffling for training. 
trainDS := baseTrainDS.BatchSize(batchSize, true).Shuffle().Infinite(true)
```
