---
title: "What is GoMLX?"
section: "Get started"
weight: 1
source: "file:///home/janpf/Projects/gomlx/gomlx/README.md"
---

# **_GoMLX_**, an Accelerated ML and Math Framework

[![GoDev](https://img.shields.io/badge/go.dev-reference-007d9c?logo=go&logoColor=white)](https://pkg.go.dev/github.com/gomlx/gomlx?tab=doc)
[![GitHub](https://img.shields.io/github/license/gomlx/gomlx)](https://github.com/gomlx/gomlx/blob/main/LICENSE)
[![Go Report Card](https://goreportcard.com/badge/github.com/gomlx/gomlx)](https://goreportcard.com/report/github.com/gomlx/gomlx)
[![Linux/amd64 Tests](https://github.com/gomlx/gomlx/actions/workflows/linux_amd64_tests.yaml/badge.svg)](https://github.com/gomlx/gomlx/actions/workflows/linux_amd64_tests.yaml)
[![Linux/arm64 Tests](https://github.com/gomlx/gomlx/actions/workflows/linux_arm64_tests.yaml/badge.svg)](https://github.com/gomlx/gomlx/actions/workflows/linux_arm64_tests.yaml)
[![Darwin/arm64 Tests](https://github.com/gomlx/gomlx/actions/workflows/darwin_tests.yaml/badge.svg)](https://github.com/gomlx/gomlx/actions/workflows/darwin_tests.yaml)
[![Windows/amd64 Tests](https://github.com/gomlx/gomlx/actions/workflows/windows_amd64_tests.yaml/badge.svg)](https://github.com/gomlx/gomlx/actions/workflows/windows_amd64_tests.yaml)
![Coverage](https://img.shields.io/badge/Coverage-67.0%25-yellow)
[![Slack](https://img.shields.io/badge/Slack-GoMLX-purple.svg?logo=slack)](https://app.slack.com/client/T029RQSE6/C08TX33BX6U)
[![Sponsor gomlx](https://img.shields.io/badge/Sponsor-gomlx-white?logo=github&style=flat-square)](https://github.com/sponsors/gomlx)

## 📖 About **_GoMLX_**
<img align="right" src="docs/gomlx_gopher2.png" alt="GoMLX Gopher" width="220px"/>

**GoMLX** is an easy-to-use set of Machine Learning and generic math libraries and tools. 
It can be seen as a **PyTorch/Jax/TensorFlow for Go**.

It can be used to train, fine-tune, modify, and combine machine learning models. 
It provides all the tools to make that work easy: from a complete set of differentiable operators, 
all the way to UItools to plot metrics while training in a notebook.

It runs almost everywhere Go runs, using a pure Go backend. 
It runs even in the browser with WASM ([see demo created with GoMLX](https://janpfeifer.github.io/hiveGo/www/hive/)). 
Likely, it will work in embedded devices as well (see [Tamago](https://github.com/usbarmory/tamago)).

It also supports a very optimized backend engine based on [OpenXLA](https://github.com/openxla/xla) 
that uses just-in-time compilation to CPU, GPUs (Nvidia, and likely AMD ROCm, Intel, Macs) and Google's TPUs.
It also supports modern distributed execution (**new, still being actively improved**) for multi-TPU or multi-GPU
using XLA Shardy, an evolution of the [GSPMD distribution](https://arxiv.org/abs/2105.04663)).

It's the same engine that powers Google's [Jax](https://github.com/google/jax), 
[TensorFlow](https://tensorflow.org/) and [Pytorch/XLA](https://docs.pytorch.org/xla/master/learn/xla-overview.html),
and it has the same speed in many cases. 
Use this backend to train large models or with large datasets.

> [!Tip]
> * See our 🎓 [**tutorial**](https://gomlx.github.io/gomlx/notebooks/tutorial.html) 🎓
> * See _Eli Bendersky_'s blog post ["GoMLX: ML in Go without Python"](https://eli.thegreenplace.net/2024/gomlx-ml-in-go-without-python/), 
>   (a bit outdated, but still useful)
> * A [guided example for Kaggle Dogs Vs Cats](https://gomlx.github.io/gomlx/notebooks/dogsvscats.html).
> * A simple [GoMLX slide deck](https://docs.google.com/presentation/d/1QWp_N9_7_n7gQKePHfmb5AFFBXnN6DTqjpWxNC0Ecpk/edit?usp=sharing) with small sample code.  

<div>
<p>It was developed to be a full-featured ML platform for Go, productionizable and easily to experiment with ML ideas
—see Long-Term Goals below.</p>

It strives to be **simple to read and reason about**, leading the user to a correct and transparent mental model 
of what is going on (no surprises)—aligned with Go philosophy.
At the cost of more typing (more verbose) at times.

It is also incredibly flexible and easy to extend and try non-conventional ideas: use it to experiment with new
optimizer ideas, complex regularizers, funky multitasking, etc.

Documentation is kept up to date (if it is not well-documented, it is as if the code is not there), 
and error messages are useful (always with a stack-trace) and try to make it easy to solve issues.
</div>

## 🔮 Upcoming Features & Plans 🔮

**Large API and package re-organization coming soon in v0.28 release:**

- `backends` moved to `github.com/gomlx/compute` repository!
  - Packages `backends`, `dtypes`, `shapes` and `distributed` moved to `github.com/gomlx/compute`.
  - The _"go"_ backend is now implemented in `github.com/gomlx/compute/gobackend` (it's been greatly improved as well).
  - The _"xla"_ backend is now implemented in `github.com/gomlx/go-xla/compute/xla`.
- Removed `pkg/` prefix from the the top level packages (no need with `internal/` special handling).
- The `context.Context` variable container redesigned (and simplified) into `model.Store` (the container)
  the `model.Scope` (a pointer to a `Store` with the current scope).

See more details in the our [CHANGELOG](docs/CHANGELOG.md), including how to use `cmd/convert_v0.28`, 
a tool that facilitates the changes needed to move to the new API -- details in the [CHANGELOG](docs/CHANGELOG.md).

**Upcoming features for the upcoming v0.28.0 release:**

* Dynamic shapes support for GoMLX: **alpha**/**experimental**, only for the Go backend (XLA only supports static shapes).
  But it will allow more efficient and padding-free (sometimes) models with Go. It also opens up better support
  for ONNXRuntime backend (a goal for after the v0.28 release).  
* Improvements and some SIMD support for the Go backend (now in repo `github.com/gomlx/compute/gobackend`).

## 🗺️ Overview

**GoMLX** is a full-featured ML framework, supporting various well-known ML components  
from the bottom to the top of the stack. But it is still only a slice of what a major ML library/framework should provide 
(like TensorFlow, Jax, or PyTorch).

### Examples developed using GoMLX

  * **🚀 NEW 🚀** [KaLM-Gema3 12B parameters](https://github.com/gomlx/go-huggingface/tree/main/examples/kalmgemma3): Tecent's top-ranked sentence encoder
    for RAGs, using [go-huggingface](https://github.com/gomlx/go-huggingface/) to load the model and tokenizer, and **GoMLX** to execute it.
  * **🚀 NEW 🚀** [Gemma 3 270M](https://github.com/gomlx/gomlx/tree/main/examples/gemma3): Demonstrates ONNX-converted
    text generation (LLM) using the [onnx-community/gemma-3-270m-it-ONNX](https://huggingface.co/onnx-community/gemma-3-270m-it-ONNX) 
    model with GoMLX. 
    It uses the [`gomlx/onnx-gomlx`](https://github.com/gomlx/onnx-gomlx) package to convert the model, and [`gomlx/go-huggingface`](https://github.com/gomlx/go-huggingface) to download the model and run the   * **🚀 NEW 🚀** [GPT-2](https://github.com/gomlx/gomlx/tree/main/examples/gpt2): Demonstrates text generation using the
    the new (experimental) transformer and generator packages.
tokenizer.
  * **🚀 NEW 🚀** [BERT-base-NER](https://github.com/gomlx/gomlx/tree/main/examples/BERT-base-NER): A BERT-base model fine-tuned
    for Named Entity Recognition. It's also a ONNX-converted model from [dslim/bert-base-NER model](https://huggingface.co/dslim/bert-base-NER) from HuggingFace.
  - **🚀 NEW 🚀** [MixedBread Reranker v1](https://github.com/gomlx/gomlx/tree/main/examples/mxbai-rerank): A cross-encoder reranking 
    example, see [HuggingFace MixedBread Reranker v1 page](https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v1).
    It uses the [`gomlx/onnx-gomlx`](https://github.com/gomlx/onnx-gomlx) package to convert the model, and [`gomlx/go-huggingface`](https://github.com/gomlx/go-huggingface) to download the model and run the tokenizer.

  * [Adult/Census model](https://gomlx.github.io/gomlx/notebooks/uci-adult.html);
  * [How do KANs learn ?](https://gomlx.github.io/gomlx/notebooks/kan_shapes.html); 
  * [Cifar-10 demo](https://gomlx.github.io/gomlx/notebooks/cifar.html); 
  * [MNIST demo (library and command-line only)](https://github.com/gomlx/gomlx/tree/main/examples/mnist)
  * [Dogs & Cats classifier demo](https://gomlx.github.io/gomlx/notebooks/dogsvscats.html); 
  * [IMDB Movie Review demo](https://gomlx.github.io/gomlx/notebooks/imdb.html); 
  * [Diffusion model for Oxford Flowers 102 dataset (generates random flowers)](examples/oxfordflowers102/OxfordFlowers102_Diffusion.ipynb);
    * [Flow Matching Study Notebook](https://gomlx.github.io/gomlx/notebooks/flow_matching.html) based on Meta's ["Flow Matching Guide and Code"](https://ai.meta.com/research/publications/flow-matching-guide-and-code/).
  * [GNN model for OGBN-MAG (experimental)](examples/ogbnmag/ogbn-mag.ipynb).
  * Last, a trivial [synthetic linear model](https://github.com/gomlx/gomlx/blob/main/examples/linear/linear.go), for those curious to see a barebones simple model.
  * Neural Style Transfer 10-year Celebration: [see a demo written using GoMLX](https://github.com/janpfeifer/styletransfer/blob/main/demo.ipynb) of the [original paper](https://arxiv.org/abs/1508.06576).
  * [Triplet Losses](https://github.com/gomlx/gomlx/blob/main/ml/train/losses/triplet.go): various negative sampling strategies as well as various distance metrics.
  * [AlphaZero AI for the game of Hive](https://github.com/janpfeifer/hiveGo/): it uses a trivial GNN to evaluate
    positions on the board. It includes a [WASM demo (runs GoMLX in the browser!)](https://janpfeifer.github.io/hiveGo/www/hive/) and a command-line UI to test your skills!

### Backends


> This page is excerpted from the [full README](https://github.com/gomlx/gomlx). For complete documentation, browse the sections in the sidebar.
