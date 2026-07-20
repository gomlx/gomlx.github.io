---
title: "Overview"
section: "Get started"
weight: 1
source: "https://github.com/gomlx/gomlx/blob/main/README.md"
---

<img align="right" src="/img/gomlx_gopher2.png" alt="GoMLX Gopher" width="220px"/>

**GoMLX** is an easy-to-use set of Machine Learning and generic math libraries and tools. 
It can be seen as a **PyTorch/Jax/TensorFlow for Go**.

It can be used to train, fine-tune, modify, and combine machine learning models. 
It provides all the tools to make that work easy: from a complete set of differentiable operators, 
all the way to UI tools to plot metrics while training in a notebook.

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

## Highlights

* **🚀 NEW 🚀**: **Gradient checkpointing**: trade-off memory usage for recomputation when training large models, with a very simple API.
* HuggingFace Go compatibility with [go-huggingface](https://github.com/gomlx/go-huggingface):
  - Download files from models/datasets sharing the same cache framework as the python version.
  - Tokenizers for various classes in pure Go, downloaded directly from HuggingFace.
  - Datasets iterators (using Apache's Parquet format), to experiment with standard datasets.
  - Model parameters reading from GGUF or `safetensors` format.
  - Model conversion to GoMLX (some models at least) with a compatible `transformer` library. Includes support to
    sentence embedding (equivalent to `sentence_transformer` Python library). 
* Convert ONNX models to GoMLX with [onnx-gomlx](https://github.com/gomlx/onnx-gomlx): both as an alternative for 
  `onnxruntime` (leveraging XLA), but also to further fine-tune models. 
* [Docker "gomlx_jupyterlab"](https://hub.docker.com/r/janpfeifer/gomlx_jupyterlab) with integrated JupyterLab 
  and [GoNB](https://github.com/janpfeifer/gonb) (a Go kernel for Jupyter notebooks)
* Autodiff: automatic differentiation—only gradients for now, no jacobian.
* `Store` and `Scope`: simple variable management for ML models.
* ML layers library with the most popular machine learning "layers": FFN layers,  
  various activation functions, layer and batch normalization, **🚀 NEW 🚀** DyT (Dynamic Tanh normalizer, https://arxiv.org/abs/2503.10622), convolutions, pooling, dropout, Multi-Head-Attention
  (for transformer layers), LSTM, KAN (B-Splines, [GR-KAN/KAT networks](https://arxiv.org/abs/2409.10594), 
  Discrete-KAN, PiecewiseLinear KAN), PiecewiseLinear (for calibration and normalization), various regularizations,
  FFT (reverse/differentiable), learnable rational functions (both for activations and [GR-KAN/KAT networks](https://arxiv.org/abs/2409.10594)),
  VNN (Vector Neural Networks) for SO(3)-Equivariant/Invariant layers, etc.
* Training library, with some pretty-printing: 
  * Plots for Jupyter notebook, using [GoNB, a Go Kernel](https://github.com/janpfeifer/gonb).
  * Various debugging tools: collecting values for particular nodes for plotting, simply logging the value
    of nodes during training, stack-trace of the code where nodes are created.
* `gomlx_checkpoints`, the command line tool to inspect checkpoint of train(-ing) models, **generate plots**
  with loss and arbitrary evaluation metrics using Plotly.
  See [example of training session](https://gomlx.github.io/gomlx/notebooks/gomlx_checkpoints_plot_example.html),
  with the effects of a learning rate change during the training.
  It also allows plotting different models together, to compare their evolution.
* Various optimizers: SGD, Adam (AdamW and Adamax).
* Various losses and metrics.
* Read Numpy arrays into GoMLX tensors -- see package `github.com/gomlx/gomlx/core/tensors/numpy`.
* **Distributed Execution** (**experimental**) across multiple GPUs or TPUs with little hints from the user.
  One only needs to configure a distributed dataset, and the trainer picks up from there.
  See code change in [UCI-Adult demo](https://github.com/gomlx/gomlx/blob/main/examples/adult/demo/main.go#L222). **Experimental**, 
  pls report any issues and help us improve it.


