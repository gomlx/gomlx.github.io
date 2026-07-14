---
weight: 80
title: "Backends & Highlights"
---
### Backends

GoMLX is a friendly "intermediary ML API", that hosts a common API and a library of ML layers and such. But per-se it doesn't execute any computation: it relies on different backends to compile and execute the computation on very different hardware.

There is a common backend interface (currently in `github.com/gomlx/gomlx/backends`, but it will soon go to its own repo), and 3 different implementations:

   1. **`xla`**: [OpenXLA](https://github.com/openxla/xla) backend for CPUs, GPUs, and TPUs. State-of-the-art as these things go, but only static-shape.
      For linux/amd64, linux/arm64 (CPU) and darwin/arm64 (CPU) for now. Using the [go-xla](https://github.com/gomlx/go-xla) Go version of the APIs.
   2. **`go`**: a pure Go backend (no C/C++ dependencies): slower but very portable (compiles to WASM/Windows/etc.): 
      * SIMD support is underway (see [SIMD for Go](https://github.com/golang/go/issues/73787) and under-development [go-highway](https://github.com/ajroetker/go-highway)); 
      * **🚀 NEW 🚀**: added support for some **fused operations** and for some types of quantization, greatly improving performance
        in some cases.
      * See also [GoMLX compiled to WASM to power the AI for a game of Hive](https://janpfeifer.github.io/hiveGo/www/hive/)
      * Dynamic shape support planned (maybe mid-2026).
   3. **🚀 NEW 🚀** **[go-darwinml](https://github.com/gomlx/go-darwinml)**: Go bindings to Apple's CoreML supporting Metal acceleration, MLX, and any backend DarwinOS related. 

