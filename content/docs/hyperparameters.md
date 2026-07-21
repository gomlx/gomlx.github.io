---
title: "Hyperparameters"
lead: "Organize, sweep, and manage model hyperparameters and command-line settings."
weight: 60
---

## Overview

In addition to stateful model variables (like weights and biases), a `model.Store` and its associated `model.Scope` hold and organize **hyperparameters** (such as learning rate, batch size, activation functions, L1/L2 regularization weight, and layer depths).

GoMLX provides a recommended (but optional) pattern for managing hyperparameters. It allows you to:
1. Define default parameter values programmatically.
2. Override parameters at runtime using a single unified command-line flag (`-set`).
3. Persist parameters inside model checkpoints, while still allowing explicit command-line overrides to take precedence when resuming training.

---

## Hierarchical Parameter Scope

Hyperparameters are organized hierarchically, mirroring the structure of your network layers. When retrieving a hyperparameter, GoMLX climbs the scope tree to find the most specific value available.

### 1. Defining Default Parameters
You can register parameters on the root store or on specific sub-scopes:

```go
store := model.NewStore()

// Set root-level defaults
store.SetParam("learning_rate", 0.001)
store.SetParam("l2_regularization", 1e-5)

// Override parameters for a specific sub-scope (e.g., output layer)
store.RootScope().In("output_layer").SetParam("l2_regularization", 0.0)
```

### 2. Retrieving Parameters inside Layers
Inside your model building functions, use `model.GetParamOr` to retrieve hyperparameters:

```go
func MyDenseLayer(scope *model.Scope, x *Node) *Node {
    // Looks for "l2_regularization" in the current scope, then parent scopes,
    // and finally falls back to 1e-5 if not configured.
    l2 := model.GetParamOr(scope, "l2_regularization", 1e-5)
    ...
}
```

---

## The Suggested CLI Integration Pattern

In large-scale machine learning, managing dozens of command-line flags for tuning can become unwieldy. The recommended GoMLX approach maps hyperparameters directly to your `model.Store` and wraps them in a single `-set` flag.

This pattern is demonstrated in the [UCI-Adult UCI demo trainer](https://github.com/gomlx/gomlx/blob/main/examples/adult/demo/main.go).

### Step 1: Define Your Default Store
Create a helper function to initialize your `model.Store` and set default hyperparameters:

```go
func createModelStore() *model.Store {
    store := model.NewStore()
    store.SetParams(map[string]any{
        "train_steps": 5000,
        "batch_size":  128,
        "optimizer":   "adam",
        "learning_rate": 0.001,
        "l2_regularization": 1e-5,
        "num_hidden_layers": 2,
    })
    return store
}
```

### Step 2: Register and Parse CLI Flags
Expose your parameters to the command line using the `github.com/gomlx/gomlx/ui/commandline` package:

```go
import (
    "flag"
    "github.com/gomlx/gomlx/ml/model"
    "github.com/gomlx/gomlx/ui/commandline"
)

func main() {
    store := createModelStore()

    // 1. Create a unified settings flag (defaults to "-set")
    settings := commandline.CreateSettingsFlag(store, "")
    
    flag.Parse()

    // 2. Parse setting overrides (returns the names of parameters that were modified)
    paramsSet, err := commandline.ParseSettings(store, *settings)
    if err != nil {
        log.Fatalf("Invalid settings: %v", err)
    }

    // 3. (Optional) Print the active settings for verification
    fmt.Println(commandline.SprintSettings(store))
    
    runTraining(store, paramsSet)
}
```

---

## Checkpoint Overrides and Precedence

When resuming training or performing evaluations from saved checkpoints, GoMLX loads the hyperparameters saved in the checkpoint directory. This ensures that you don't have to re-enter network architectures or training settings.

However, some flags must behave differently:
1. **Current Run Context**: Parameters like `train_steps` or whether to generate `plots` belong to the current execution, not the saved history.
2. **Explicit CLI Overrides**: If you run `-set "learning_rate=0.0001"`, you want your command-line override to take precedence over the saved checkpoint learning rate.

You handle this precedence by passing excluded parameters to the `checkpoint.Build` handler:

```go
import "github.com/gomlx/gomlx/ml/model/checkpoint"

// Define parameters that should always use current settings
alwaysCurrentParams := []string{"train_steps", "plots"}

// Exclude both explicitly set parameters (from command line) and current context params
excludedParams := append(paramsSet, alwaysCurrentParams...)

checkpointHandler, err := checkpoint.Build(store).
    Dir("/path/to/checkpoints").
    ExcludeParams(excludedParams...). // Keep these from being overwritten by the checkpoint
    Done()
```

---

## CLI `-set` Flag Format

The `-set` flag supports multiple formats for flexibility:

### 1. Semicolon-Separated Overrides
Override multiple parameters directly:
```bash
go run ./cmd/train -set "learning_rate=0.005;batch_size=256"
```

### 2. Scoped Overrides
Override parameters within a specific hierarchical path:
```bash
go run ./cmd/train -set "/output_layer/l2_regularization=0.0;learning_rate=0.01"
```

### 3. Loading from a Settings File
For complex configurations or hyperparameter sweeps, you can load settings from a file:
```bash
go run ./cmd/train -set "file:configs/hparams_sweep_3.txt"
```

**Inside `configs/hparams_sweep_3.txt`**:
```ini
# Settings file example
learning_rate=0.0005
batch_size=512
l2_regularization=1e-6
# Disable dropout for output
/output_layer/dropout=0.0
```
