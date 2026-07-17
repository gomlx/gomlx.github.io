---
title: "Error Handling"
lead: "Understand GoMLX's error handling design, graph building exceptions, and helper patterns."
weight: 60
---

## Overview

GoMLX uses a hybrid approach to error handling to balance Go's safety with mathematical readability:

1. **Graph Building Time (Exceptions)**: Operations building the computation graph (e.g. `layers.Dense`, `graph.Add`, `nn.Relu`) panic with a stack-trace error on invalid shapes, mismatched data types, or invalid configurations.
2. **Graph Execution and Ecosystem (Standard Errors)**: Running compiled graphs, loading datasets, reading files, or saving checkpoints return standard Go `(value, error)` pairs. 
3. **`Must`-prefixed Conveniences**: For scripting and main files, standard error-returning functions often have a `Must` version (e.g. `MustCall`, `MustNewExec`) that converts errors into panics automatically.

---

## Why Graph Building Panics

In idiomatic Go, checking `if err != nil` at every step is the standard. However, deep learning model building consists of long chains of algebraic compositions. Checking errors on every mathematical operation would make model definitions unreadable.

### Comparison: Euclidean Distance Formula

Consider the standard Euclidean distance formula: $d = \sqrt{\sum (x - y)^2}$

**Without exceptions (standard Go):**
```go
// Every math operation (Sub, Square, Reduce, Sqrt) could fail due to shape/type mismatch.
diff, err := Sub(x, y)
if err != nil {
    return nil, err
}
squared, err := Square(diff)
if err != nil {
    return nil, err
}
sum, err := ReduceAllSum(squared)
if err != nil {
    return nil, err
}
distance, err := Sqrt(sum)
if err != nil {
    return nil, err
}
```

**With exceptions (GoMLX):**
```go
l2 := ReduceAllSum(Square(Sub(x, y)))
distance := Sqrt(l2)
```

The exception-based builder is clean, readable, and directly mirrors the mathematical definition of Euclidean distance.
It allows developers to focus on the formulas without boilerplate.

---

## Mitigating Exception Downsides

While panics are generally discouraged in most Go applications, model graph building fits well with exceptions because:

* **Sequential Setup**: Graph construction is sequential (no goroutines). There are no race conditions or background goroutines left in unstable states during a panic.
* **No Performance Impact**: Graph building is done only once when initializing/warming-up the model. The graph is then JIT-compiled. During the training loop execution, no panics are used.
* **Panics Are Handled**: The executors (`graph.Exec` and `model.Exec`) capture the panics and handle them cleanly.
  So in most use cases, the one doen't need to handle them.

---

## The `gomlx/support/exceptions` Package

GoMLX defines a lightweight exception package located at `github.com/gomlx/gomlx/support/exceptions`.

### 1. Thow Exception: `Panicf`
Use `exceptions.Panicf` inside layers or custom operators to format an error message and panic with a stack trace:

```go
import "github.com/gomlx/gomlx/support/exceptions"

func Dense(scope *model.Scope, input *Node, dimensions int) *Node {
    if input.Shape().Rank() < 2 {
        // Automatically wraps the error message with a full stack trace and panics
        exceptions.Panicf("fnn: input must be rank at least 2, got shape=%s", input.Shape())
    }
    ...
}
```

### 2. Capture Exceptions: `TryCatch` and `Try`
Mostly used for internal development, simplifies capturing compilation boundaries using `exceptions.TryCatch`:

```go
import "github.com/gomlx/gomlx/support/exceptions"

func compileAndRun() (outputs []*tensors.Tensor, err error) {
    // TryCatch catches only panics of type E (in this case error) 
    // and returns them. Other unexpected panic types are re-thrown.
    err = exceptions.TryCatch[error](func() {
        graph := graph.NewGraph(backend, "inference")
        x := graph.Parameter("x", shape)
        y := BuildMyComplexNetwork(graph, x)
        
        exec := graph.Compile(y)
        outputs = exec.Call(input)
    })
    return
}
```

---

## The `Must` Prefix Convention

For helper functions that interact with the host system (such as tensors, compiled execution, and file storage), GoMLX provides both standard error-returning functions and panic-on-failure variants:

| Standard Method (Returns error) | `Must` Method (Panics on error) |
| :--- | :--- |
| `graph.Compile(nodes...)` | `graph.MustCompile(nodes...)` |
| `exec.Call(inputs...)` | `exec.MustCall(inputs...)` |
| `tensors.FromFlatData(...)` | `tensors.MustFromFlatData(...)` |
| `checkpointHandler.Save()` | `checkpointHandler.MustSave()` |

Use the standard error-returning methods when writing production APIs or libraries, and use the `Must` functions inside 
test cases, CLI tools, or `main()` scripts to keep code flow straight.
