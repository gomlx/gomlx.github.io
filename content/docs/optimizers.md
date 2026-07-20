---
title: "Optimizer"
section: "Training & Layers"
weight: 29
---

Optimizers in GoMLX are responsible for updating the model variables (weights and biases) to minimize a scalar loss function. All optimizers implement the `optimizer.Interface` and are designed to work seamlessly with GoMLX's JIT graph compilation.

---

## High-Level Integration: `train.Trainer`

In most cases, you do not interact with the optimizer directly. Instead, you create it and pass it to the constructor of `train.Trainer`:

```go
import (
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/optimizer"
)

// Construct trainer with an Adam optimizer configured from the scope/store
trainer := train.NewTrainer(
	backend,
	store,
	modelFn,
	lossFn,
	optimizer.Adam().LearningRate(0.001).Done(),
	trainMetrics,
	evalMetrics,
)
```

You can also use `optimizer.FromStore(store)` to dynamically construct the optimizer based on parameters configured in the model store. This is extremely convenient when doing hyperparameter search or overriding options via command line flags (e.g., `-set=optimizer=adamw -set=learning_rate=0.0003`).

---

## Direct Usage in Custom Training Loops

If you are building a custom training pipeline, you can use the optimizer directly inside your step graph.

### The `optimizer.Interface`
```go
type Interface interface {
	// UpdateGraph calculates the updates to the variables of the model for one step.
	// The loss must be a scalar value.
	UpdateGraph(scope *model.Scope, g *Graph, theLoss *Node)

	// Clear deletes all temporary variables (moments, step counters) used by the optimizer.
	Clear(scope *model.Scope) error
}
```

Because GoMLX variables are stateful and JIT-compiled, calling `UpdateGraph` registers the weight updates directly on the symbolic graph variables using `Variable.SetNodeValue`. 

When the executor runs, it automatically materializes the updated weights and writes them back to the `model.Store` in-place. You do not need to extract variables or apply gradients manually.

### Custom Training Loop Example:
```go
import (
	"fmt"
	"log"

	"github.com/gomlx/compute"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/train/loss"
	"github.com/gomlx/gomlx/ml/train/optimizer"
)

func main() {
	backend := compute.MustNew()
	store := model.NewStore()
	opt := optimizer.Adam().LearningRate(0.001).Done()

	// 1. Define the training step graph function
	trainStepFn := func(scope *model.Scope, x, y *Node) *Node {
		// A. Forward pass
		predictions := myModelFn(scope, x)
		// B. Scalar loss
		lossNode := loss.MeanSquaredError(y, predictions)
		// C. Register gradient updates in the graph
		opt.UpdateGraph(scope, x.Graph(), lossNode)
		return lossNode
	}

	// 2. Build the execution graph
	exec := model.NewExec(backend, store, trainStepFn)

	// 3. Iterate in your custom training loop
	for step := 0; step < 1000; step++ {
		inputs, labels := getNextBatch() // Yields *tensors.Tensor inputs and labels

		// Executing the graph automatically updates the variable tensors in model.Store
		results, err := exec.Call(inputs, labels)
		if err != nil {
			log.Fatalf("failed step %d: %v", step, err)
		}
		fmt.Printf("Step %d Loss: %v\n", step, results[0].Value())

		// Crucial: finalize returned tensors to avoid device VRAM leaks
		results[0].MustFinalizeAll()
		inputs.MustFinalizeAll()
		labels.MustFinalizeAll()
	}
}
```

---

## Stochastic Gradient Descent (SGD)

Created using `optimizer.StochasticGradientDescent()`. SGD updates variables by moving them in the opposite direction of the gradient scaled by the learning rate.

* **Learning Rate Decay**: By default, it features a square-root decay schedule:
  $$\text{learning\_rate} = \frac{\text{initial\_lr}}{\sqrt{\text{global\_step}}}$$
  You can disable this decay schedule using `.WithDecay(false)`.

```go
opt := optimizer.StochasticGradientDescent().
	WithLearningRate(0.1).
	WithDecay(false). // Disable default square-root decay
	Done()
```

---

## Adam & Variants

Created using `optimizer.Adam()`. The Adam optimizer adaptively calculates learning rates for each parameter using estimations of the first (momentum) and second (uncentered variance) moments of the gradients.

GoMLX implements several widely used extensions of the Adam optimizer:

### 1. Standard Adam
Tracks running moments to adjust gradients dynamically.
* **Paper**: [Adam: A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980) (Kingma et al., 2014)

### 2. AdamW
Incorporate weight decay directly into the weight update step rather than scaling gradient L2 regularization (which doesn't perform well with adaptive momentums).
* **Paper**: [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) (Loshchilov & Hutter, 2017)
* **Usage**: `optimizer.Adam().WeightDecay(0.004).Done()` (or setting the `"adam_weight_decay"` hyperparameter).

### 3. Adamax
A variant of Adam based on the infinity norm ($L_\infty$) for the second moment, providing additional numerical stability.
* **Paper**: Described in the original Adam paper (Kingma et al., 2014).
* **Usage**: `optimizer.Adam().Adamax().Done()`

### 4. AMSGrad
Uses the maximum of past squared gradients instead of exponential decays, guaranteeing convergence under certain settings where standard Adam fails.
* **Paper**: [On the Convergence of Adam and Beyond](https://arxiv.org/abs/1904.09237) (Reddi et al., 2018).
* **Usage**: `optimizer.Adam().AMSGrad().Done()`

### 5. RMSProp
Divides the learning rate by a running average of recent gradient magnitudes (RMS) without computing momentum (first moment).
* **Paper**: Hinton's Coursera Lecture, and Graves' [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/abs/1308.0850).
* **Usage**: `optimizer.RMSProp().Done()`
