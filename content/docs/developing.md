---
title: "Developing"
section: "Guides"
weight: 40
source: "file:///home/janpf/Projects/gomlx/gomlx/docs/developing.md"
---

Below is a list of usual low level implementation tasks:

## Updating `coverage.out` file

This is not done as a GitHub actions because it would take too long to download the datasets, etc.
Instead, we do it manually using the `cmd/run_coverage.sh` simple script. 

It takes some 10-20 minutes to run, and updates the file `docs/converage.out`.
Once the file is submitted, a GitHub action will update the coverage badge.

## Adding new backend operation

Here we are referring to new operations defined in the backend, as opposed to new operations that are
combinations of what already exist.

The main backend is `xla`, provided by the `github.com/gomlx/gopjrt` repository. 
You want to add the op there, first, then in the `xla` package. 
See the various generators under `cmd/` (configured with `go:generate`): you want to either use them,
or exclude the new op from using them. It's always a good practice to re-generate (`go generate ./...`)
and see the difference.

## Adding support for a new DType

**GoMLX** uses dtypes defined in `gopjrt` repository.
The `gopjrt/dtypes` package provides lots of support functionality (lots of generics) in order to
make it simple to add new data types. 

If adding new data types, consider adding tests also in the package `tensors`, to make sure the conversions back and
forth are working.

---

## Compile time, Runtime, **Graph Building Time** and **Graph Execution Time**

This section is just some historical notes on how we got to this design based on building a computation graph in GoMLX.

Strong type checking is a major plus for a programming language. It allows catching mistakes much earlier:
in compile time or even IDEs can catch them, while editing. And in some cases it allows catching errors
that would go unseen by testing (odd combination of values uncovered by test). 

Unfortunately, it is not possible to encode Graph type checking in compile time, it would require extensions
to the compiler, and in some cases it actually requires running the code, since the output of some ops tensors
have shapes that are dynamically generated.

This slows down development, since it requires running the programs to check the validity their computation
graph types and values.

One recommendation here is to execute the graph building (and not necessarily execution) early in the 
program execution -- before data preprocessing for instance. So this can be iterated fast.

Now there are issues that happen (and can be caught) during graph building
and others that happen only during graph execution (e.g.: NaNs in computation) later on. 
One can think that the graph building code is being executed in two different modes, 
and for convenience we often split the concept of "runtime" for those
into **"graph building time"** and **"graph execution time"**.

> **Note**: This is not as big an issue in practice as some characterize it. As an anecdote TensorFlow 2.0
> adoption of "eager execution mode" (same as PyTorch), to hide the graph building behind the scenes, caused more problems than
> it solved -- developing models in TensorFlow 1.0 was generally simpler and faster
> (in terms of developer speed). But this is a longer topic (and confounded by other aspects that were
> improved in TensorFlow 2.0/Keras).
