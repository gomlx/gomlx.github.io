---
weight: 60
title: "go-huggingface"
---

# **go-huggingface**: Download, Tokenize, and Import HuggingFace Models

The [**go-huggingface**](https://github.com/gomlx/go-huggingface) library provides simple, clean APIs in Go to interact with the [HuggingFace Hub](https://huggingface.co). It includes packages for **downloading** files (`hub`), **tokenizing** text (`tokenizers`), **streaming datasets** (`datasets`), and **importing model weights** directly into GoMLX graphs (`models/transformer`).

{{< callout type="note" >}}
**No GoMLX Dependency for Core Features**: Packages like `hub`, `tokenizers`, and `datasets` are fully self-contained and **do not depend on GoMLX**. You can use them to download files, tokenize sentences, or stream Parquet datasets in any standard Go project.
{{< /callout >}}

---

## 1. Downloading from HuggingFace Hub (`hub`)

**Package**: `github.com/gomlx/go-huggingface/hub`

The `hub` package provides information from any model or dataset repository on the Hub and downloads files. It shares the same cache directory format as the official HuggingFace Python client, so if you already downloaded a model via Python, Go will reuse the same cached files (avoiding duplicate downloads).

### Usage Example
```go
import (
	"fmt"
	"os"

	"github.com/gomlx/compute/support/humanize"
	"github.com/gomlx/go-huggingface/hub"
)

func main() {
	hfAuthToken := os.Getenv("HF_TOKEN") // Optional token for private or gated models
	repo := hub.New("google/gemma-2-2b-it").WithAuth(hfAuthToken)

	// Iterate over files in the repository
	for fileInfo, err := range repo.IterFileInfos() {
		if err != nil {
			panic(err)
		}
		fmt.Printf("File: %s (Size: %s)\n", fileInfo.Name, humanize.Bytes(fileInfo.Size))
	}

	// Download a specific file to the cache
	localPath, err := repo.DownloadFile("config.json")
	if err != nil {
		panic(err)
	}
	fmt.Printf("Downloaded config.json to: %s\n", localPath)
}
```

---

## 2. Text Tokenizers (`tokenizers`)

**Package**: `github.com/gomlx/go-huggingface/tokenizers`

The `tokenizers` package provides a generic `Tokenizer` interface and multiple implementations:
1. **SentencePiece (Go-native)**: A pure Go tokenizer implementation used by Gemma and Gemma-2 models.
2. **Rust bindings (`github.com/daulet/tokenizers`)**: Integrates with the high-performance Rust tokenizer library to support WordPiece, BPE, and other tokenizers.

### Example: Tokenizing Gemma Text
```go
import (
	"fmt"
	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/tokenizers"
)

func main() {
	repo := hub.New("google/gemma-2-2b-it")
	tokenizer, err := tokenizers.New(repo)
	if err != nil {
		panic(err)
	}

	sentence := "The book is on the table."
	tokens := tokenizer.Encode(sentence)
	fmt.Printf("Tokens: %v\n", tokens) // Output: [651 2870 603 611 573 3037 235265]
}
```

### Sentence Bucketizing (`tokenizers/bucket`)
When processing batches of text for deep learning, padding sentences to the maximum length of the batch wastes memory and compute. The `tokenizers/bucket` package bucketizes sentences of similar lengths together (using strategies like Power-of-2, Power-of-X, or Two-Bits), dynamically padding only what is necessary:

```go
import "github.com/gomlx/go-huggingface/tokenizers/bucket"

// Create a bucketizer that batches sentences targeting ~8k tokens per bucket
bkt := bucket.New(tokenizer).
	ByTwoBitBucketBudget(8*1024, 16).
	WithMaxParallelization(-1)

// Write sentences to bucketsInputChan; read batched, padded buckets from bucketsOutputChan
go bkt.Run(bucketsInputChan, bucketsOutputChan)
```

---

## 3. Streaming Datasets (`datasets`)

**Package**: `github.com/gomlx/go-huggingface/datasets`

The `datasets` package downloads and streams Parquet-based HuggingFace datasets efficiently. It contains command-line helper tools to fetch and process datasets:

* **`dataset_download`**: Downloads, lists, or deletes split dataset files.
  ```bash
  go run github.com/gomlx/go-huggingface/cmd/dataset_download microsoft/ms_marco
  ```
* **`generate_dataset_structs`**: Generates Go structs matching the Parquet schema.
  ```bash
  go run github.com/gomlx/go-huggingface/cmd/generate_dataset_structs -dataset HuggingFaceFW/fineweb
  ```

### Example: Streaming Parquet Records
Using the generated struct, you can stream records page-by-page. Files are downloaded on-demand in the background, minimizing memory overhead:

```go
import "github.com/gomlx/go-huggingface/datasets"

type FinewebRecord struct {
	Text          string  `parquet:"text"`
	ID            string  `parquet:"id"`
	LanguageScore float64 `parquet:"language_score"`
}

func main() {
	ds := datasets.New("HuggingFaceFW/fineweb")
	
	// Stream records lazily
	for record, err := range datasets.IterParquetFromDataset[FinewebRecord](ds, "sample-10BT", "train") {
		if err != nil {
			panic(err)
		}
		fmt.Printf("ID: %s, Score: %.3f\n", record.ID, record.LanguageScore)
	}
}
```

---

## 4. Importing Transformer Models in GoMLX (`models/transformer`)

**Package**: `github.com/gomlx/go-huggingface/models/transformer`

> [!WARNING]
> **EXPERIMENTAL**: This feature is currently in development and works primarily for Gemma and BERT-based models.

The `models/transformer` package parses HuggingFace JSON configurations and safetensors weight files, mapping the model parameters directly into a GoMLX `model.Store`. It can then build the JIT-compiled attention graph for execution:

```go
import (
	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/models/transformer"
	"github.com/gomlx/gomlx/ml/model"
)

func main() {
	repo := hub.New("tencent/KaLM-Embedding-Gemma3-12B-2511")
	hfModel, err := transformer.LoadModel(repo)
	if err != nil {
		panic(err)
	}

	// Load model weights into the GoMLX store
	store := model.NewStore()
	err = hfModel.LoadStore(backend, store)

	// Build the computation graph
	exec, err := model.NewExec1(backend, store, func(scope *model.Scope, tokens *graph.Node) *graph.Node {
		// Sentence embedding graph
		return hfModel.SentenceEmbeddingGraph(scope, tokens, nil)
	})
}
```

---

## 5. Examples Catalog

The repository includes several complete examples under the `examples/` directory showing how to use these packages:

* [**`BAAI-bge-small-en-v1.5`**](https://github.com/gomlx/go-huggingface/tree/main/examples/BAAI-bge-small-en-v1.5):
  Demonstrates how to run a BERT-based small, high-performance sentence embedder model (English v1.5) to perform similarity scoring between query and document vectors in GoMLX.
* [**`gemma4-e4bit`**](https://github.com/gomlx/go-huggingface/tree/main/examples/gemma4-e4bit):
  Demonstrates how to execute text generation using Google's quantized Gemma-4-E4B-it (4-bit) model, employing a key-value (KV) cache for fast autoregressive generation.
* [**`kalmgemma3`**](https://github.com/gomlx/go-huggingface/tree/main/examples/kalmgemma3):
  Shows how to load and execute Tencent's KaLM-Embedding-Gemma3-12B-2511 Sentence Encoder for dense vector sentence representation and similarity calculations.
* [**`msmarco`**](https://github.com/gomlx/go-huggingface/tree/main/examples/msmarco):
  Loads and iterates over Microsoft's MS MARCO passage ranking dataset. It features `benchmark_embed`, a CLI benchmarking binary that measures embedding throughput (passages per second) across various compute backends.