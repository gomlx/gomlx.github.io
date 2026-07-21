---
title: "ONNX-GoMLX"
weight: 70
---

# **ONNX-GoMLX**: Inference and Fine-Tuning ONNX Models with GoMLX

The [**ONNX-GoMLX**](https://github.com/gomlx/onnx-gomlx) library allows Go developers to load and convert [ONNX models](https://onnx.ai/) (`.onnx`) directly into [GoMLX](https://github.com/gomlx/gomlx) computation graphs. 

This enables hardware-accelerated inference (CPU/GPU/TPU via GoMLX's XLA backend) without needing Python or the C++ ONNX Runtime library, as well as full **fine-tuning** of pre-trained ONNX models using GoMLX's automatic differentiation and training loops.

---

## Key Features & Use Cases

1. **Inference Without ONNX Runtime or Python**: Execute ONNX models natively in Go via GoMLX and XLA. You can easily wrap pre- and post-processing logic (such as image normalization, tokenization, or custom output heads) directly in Go graph nodes around the ONNX model.
2. **Fine-Tuning Pre-Trained Models**: Import model weights into a GoMLX `model.Store`, use GoMLX's `train.Trainer` to fine-tune on custom datasets, and save the updated weights as GoMLX checkpoints or export them back to an `.onnx` file.
3. **Zero Python Runtime Dependencies**: Production binaries only depend on Go code and the GoMLX compute backend.

---

## 1. Loading and Inspecting an ONNX Model

**Packages**: `github.com/gomlx/onnx-gomlx/onnx` and `github.com/gomlx/onnx-gomlx/onnx/parser`

You can load an ONNX model file using `onnxparser.FromFile(path)` or `onnx.ReadFile(path)`:

```go
import (
	"fmt"
	"github.com/gomlx/onnx-gomlx/onnx"
	"github.com/gomlx/onnx-gomlx/onnx/parser"
)

func main() {
	// Parse the ONNX model file
	onnxModel, err := onnxparser.FromFile("path/to/model.onnx")
	if err != nil {
		panic(err)
	}
	defer onnxModel.Close()

	// Inspect graph inputs and outputs
	inputNames, inputShapes := onnxModel.Inputs()
	outputNames, outputShapes := onnxModel.Outputs()

	fmt.Printf("Model Inputs:  %v (%v)\n", inputNames, inputShapes)
	fmt.Printf("Model Outputs: %v (%v)\n", outputNames, outputShapes)
}
```

---

## 2. Running Inference with GoMLX

To run inference, you transfer the model's weights into a GoMLX `model.Store` using `VariablesToScope`, and build a computation graph with `CallGraph`.

### Complete Inference Example
The example below uses `go-huggingface` to download the `all-MiniLM-L6-v2` sentence embedding ONNX model and executes it using GoMLX:

```go
package main

import (
	"fmt"
	"os"

	"github.com/gomlx/compute"
	_ "github.com/gomlx/gomlx/backends/default" // Import default backend (XLA)
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/onnx-gomlx/onnx"
	"github.com/gomlx/onnx-gomlx/onnx/parser"
	"github.com/gomlx/gomlx/ml/model"
)

func main() {
	// 1. Download and cache the ONNX model from HuggingFace
	repo := hub.New("sentence-transformers/all-MiniLM-L6-v2").WithAuth(os.Getenv("HF_TOKEN"))
	modelPath, err := repo.DownloadFile("onnx/model.onnx")
	if err != nil {
		panic(err)
	}

	// 2. Parse ONNX model
	onnxModel, err := onnxparser.FromFile(modelPath)
	if err != nil {
		panic(err)
	}
	defer onnxModel.Close()

	// 3. Load weights into GoMLX Store
	store := model.NewStore()
	if err := onnxModel.VariablesToScope(store.RootScope()); err != nil {
		panic(err)
	}

	// 4. Sample token inputs (shape [batch=2, seq_len=7])
	inputIDs := tensors.FromValue([][]int64{
		{101, 2023, 2003, 2019, 2742, 6251, 102},
		{101, 2169, 6251, 2003, 4991, 102, 0},
	})
	tokenTypeIDs := tensors.FromValue([][]int64{
		{0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0},
	})
	attentionMask := tensors.FromValue([][]int64{
		{1, 1, 1, 1, 1, 1, 1},
		{1, 1, 1, 1, 1, 1, 0},
	})

	// 5. Execute computation graph with GoMLX
	backend := compute.MustNew()
	results := model.MustCallOnceN(
		backend,
		store,
		func(scope *model.Scope, inputs []*Node) []*Node {
			// Convert ONNX graph nodes to GoMLX nodes
			return onnxModel.CallGraph(scope, inputs[0].Graph(), map[string]*Node{
				"input_ids":      inputs[0],
				"attention_mask": inputs[1],
				"token_type_ids": inputs[2],
			})
		},
		inputIDs, attentionMask, tokenTypeIDs,
	)

	fmt.Printf("Output Embeddings Shape: %s\n", results[0].Shape())
	fmt.Printf("Embeddings Value:\n%s\n", results[0])
}
```

---

## 3. Fine-Tuning and Exporting Back to ONNX

You can fine-tune an imported ONNX model using standard GoMLX training loops (`train.Trainer`), and then write the updated parameters back into the ONNX model file.

### Workflow:
1. **Load Weights**: Call `onnxModel.VariablesToScope(store.RootScope())` to populate `model.Store`.
2. **Build Graph**: Wrap `onnxModel.CallGraph(...)` inside your `modelFn` passed to `train.NewTrainer(...)`.
3. **Train**: Run standard training loops using `loop.RunSteps(...)` or `loop.RunEpochs(...)`.
4. **Sync Back to ONNX**: Call `onnxModel.ScopeToONNX(store.RootScope())` to copy updated variable values from GoMLX back into the in-memory ONNX protocol buffer.
5. **Save File**: Call `onnxModel.SaveToFile("fine_tuned_model.onnx")` or `onnxModel.Write(writer)`.

```go
// 1. Train the model using GoMLX trainer...
trainer := train.NewTrainer(backend, store, modelFn, lossFn, optimizer, trainMetrics, evalMetrics)
loop := train.NewLoop(trainer)
loop.RunEpochs(trainDataset, 5)

// 2. Copy updated weights from GoMLX Store back to the ONNX model structure
if err := onnxModel.ScopeToONNX(store.RootScope()); err != nil {
	panic(err)
}

// 3. Export the fine-tuned ONNX model to disk
if err := onnxModel.SaveToFile("fine_tuned_model.onnx"); err != nil {
	panic(err)
}
fmt.Println("Successfully exported fine-tuned ONNX model!")
```

---

## 4. Configuration & Advanced Options

The `onnx.Model` interface provides options to customize graph conversion:

* **`AllowDTypePromotion()`**: Automatically converts and promotes scalar/tensor types for operations with mismatched dtypes.
* **`PrioritizeFloat16()`**: Prefers `Float16` over `Float32` during dtype promotion (useful for memory optimization on GPU/TPU).
* **`WithInputsAsConstants(map[string]any)`**: Marks specific graph inputs as constant values, allowing the compiler to optimize and fold constant expressions.
* **`FreeUnusedVariables()`**: Releases memory for variables present in the ONNX file that are not actually consumed by the computation graph.
* **`WithBaseDir(path)`**: Sets the base directory for resolving external `.weight` or `.data` files referenced by large ONNX models.

---

## 5. Coverage and Benchmarks

GoMLX implements conversion for most common ONNX operators used in computer vision (CNNs, ResNets, InceptionV3), NLP (Transformers, BERT, MiniLM, DeBERTa), and tabular models.

### Benchmarks
In benchmark tests on BERT-based sentence encoders (such as `all-MiniLM-L6-v2`), GoMLX's XLA backend achieves performance comparable to Microsoft's C++ ONNX Runtime on CPU and GPU, while significantly outperforming pure-Go CPU interpreters.
