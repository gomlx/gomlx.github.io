---
title: "Loss"
section: "Training & Layers"
weight: 31
---

Loss functions in GoMLX measure the difference between model predictions and target labels. The model's parameters are updated during training to minimize the value output by the loss function. All standard and custom loss functions follow the `train.LossFn` type.

---

## High-Level Integration: `train.Trainer`

A loss function is typically passed as the fourth argument to `train.NewTrainer(...)`:

```go
import (
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/loss"
)

trainer := train.NewTrainer(
	backend,
	store,
	modelFn,
	loss.BinaryCrossentropyLogits, // Pass predefined loss function
	optimizer,
	trainMetrics,
	evalMetrics,
)
```

During training and evaluation steps, the trainer executes the loss function on the model's predictions and target labels. The trainer automatically wraps the returned loss node in `graph.ReduceAllMean(...)`, reducing any multi-dimensional tensor to a scalar before backpropagation.

---

## Defining Custom Losses

You can easily define your own loss function by matching the `train.LossFn` signature:

```go
type LossFn func(labels, predictions []*Node) *Node
```

### Example: Custom Mean Squared Error
```go
import . "github.com/gomlx/gomlx/core/graph"

func CustomMSE(labels, predictions []*Node) *Node {
	yTrue := labels[0]
	yPred := predictions[0]
	
	// Compute mean squared error: Mean((yTrue - yPred)^2)
	squaredDiff := Square(Sub(yTrue, yPred))
	return ReduceMean(squaredDiff)
}
```

### Multi-Head Output Losses
Predefined losses in `ml/train/loss` assume predictions and labels are slices of length 1. For multi-head models returning multiple prediction tensors, you can write a custom `LossFn` to split the slices, delegate to predefined losses, and sum the results:

```go
func MultiHeadLoss(labels, predictions []*Node) *Node {
	// First head: classification (logits)
	classLoss := loss.SparseCategoricalCrossEntropyLogits(labels[0:1], predictions[0:1])
	
	// Second head: bounding box regression (MSE)
	bboxLoss := loss.MeanSquaredError(labels[1:2], predictions[1:2])
	
	// Combine with weights
	return Add(classLoss, MulScalar(bboxLoss, 0.5))
}
```

---

## Predefined Losses

The `github.com/gomlx/gomlx/ml/train/loss` package provides multiple standard losses, categorized below.

### 1. Regression Losses

* **`loss.MeanAbsoluteError` (MAE)**: Calculates the average absolute differences ($L_1$ loss). More robust to outliers.
* **`loss.MeanSquaredError` (MSE)**: Calculates the average squared differences ($L_2$ loss). Penalizes larger errors heavily.
* **`loss.MakeHuberLoss(delta float64)`**: A smooth hybrid loss. It behaves like MSE for small errors ($< \delta$) and MAE for large errors, combining the advantages of both.
  * *Usage*: `loss.MakeHuberLoss(1.0)`
* **`loss.MakeAdaptivePowerLoss(p float64)`**: A generalized loss of the form $|y_{\text{true}} - y_{\text{pred}}|^p$. When $p=1$ it is MAE, and $p=2$ is MSE. Setting $p$ as a learnable parameter allows the model to adaptively find the optimal norm.
  * *Usage*: `loss.MakeAdaptivePowerLoss(1.5)`

### 2. Classification Losses

* **`loss.BinaryCrossentropy`**: Crossentropy for binary classifications ($\in [0, 1]$). Expects probability inputs.
* **`loss.BinaryCrossentropyLogits`**: Crossentropy for binary classifications where inputs are raw log-odds (logits). This is numerically stable and preferred over probability inputs.
* **`loss.CategoricalCrossEntropy`**: Crossentropy for multi-class classification. Expects probability distributions (e.g. from Softmax).
* **`loss.CategoricalCrossEntropyLogits`**: Crossentropy for multi-class classification where inputs are raw logits.
* **`loss.SparseCategoricalCrossEntropyLogits`**: Multi-class crossentropy where target labels are integer indices (instead of one-hot encoded probability vectors) and predictions are raw logits.

### 3. Distance & Metric Learning Losses

* **`loss.EuclideanDistance`**: Standard $L_2$ distance.
* **`loss.EuclideanDistanceSquare`**: Squared $L_2$ distance.
* **`loss.TripletLoss`**: Used for representation/metric learning (such as Siamese networks). It takes anchor, positive, and negative embeddings, encouraging positive pairs to be close and negative pairs to be far apart by at least a specified `margin`:
  * *Signature*: `loss.TripletLoss(labels, predictions, margin, strategy, metric)`
  * *Mining Strategies*: Supports `loss.TripletMiningBatchHard` (mining the hardest positives and negatives within the batch) and `loss.TripletMiningBatchAll`.
  * *Distance Metrics*: Supports `loss.PairwiseDistanceEuclidean`, `loss.PairwiseDistanceSquaredEuclidean`, and `loss.PairwiseDistanceCosine`.

---

## Dynamic Loss Configuration

You can fetch loss functions dynamically from the model scope's hyperparameters using `loss.LossFromScope(scope)`:

```go
// Setup your training params:
store.SetParams(map[string]any{
	loss.ParamLoss: "binary_crossentropy_logits",
})

// Create loss from scope:
lossFn, err := loss.LossFromScope(scope)
```
Valid string values for `ParamLoss` include `"mae"`, `"mse"`, `"huber"`, `"apl"`, `"binary_crossentropy"`, `"binary_crossentropy_logits"`, `"categorical_crossentropy"`, `"categorical_crossentropy_logits"`, `"sparse_categorical_crossentropy_logits"`, `"triplet"`, `"euclidean"`, and `"euclidean_square"`.

---

## Extra Loss Accumulation in `train.Trainer`

In some unconventional model architectures, the target loss is not simply calculated from predictions and labels at the end of the forward pass. Instead, various intermediate layers may generate loss terms (e.g. regularization penalties on activations, or latent space alignment terms).

GoMLX supports this by allowing you to add loss terms *anywhere* inside the model function during graph construction:

* **`train.AddMainLoss(theLoss *graph.Node)`**: Accumulates a loss term as part of the main optimization objective.
* **`train.AddRegularizationLoss(theLoss *graph.Node)`**: Accumulates a loss term as a regularization penalty (e.g., L1/L2 weight decay).

These functions can be called multiple times. GoMLX automatically sums all main and regularization terms into a single total loss minimized by the optimizer, but reports the main loss and regularization loss values separately in training logs and plots.

### Unsupervised / Self-Supervised Training Example

In self-supervised architectures (such as BYOL or Contrastive learning), there is no simple label vector passed from a dataset. In this case, you can set the trainer's `lossFn` to `nil` and accumulate the main loss directly within your model function:

```go
import "github.com/gomlx/gomlx/ml/train"

// In your training setup, pass nil for the loss function:
trainer := train.NewTrainer(backend, store, modelFn, nil, optimizer, metrics, evalMetrics)

// In your model function, compute and register the loss:
func MySelfSupervisedModel(scope *model.Scope, inputs []*Node) []*Node {
	g := inputs[0].Graph()
	representation1 := branch1(scope.In("branch1"), inputs[0])
	representation2 := branch2(scope.In("branch2"), inputs[0])

	// Calculate a custom contrastive loss
	contrastiveLoss := customContrastiveDistance(representation1, representation2)

	// Register it as the main loss.
	// Since no lossFn was passed to NewTrainer, the trainer uses this term to optimize.
	train.AddMainLoss(contrastiveLoss)

	return []*Node{representation1}
}
```

{{% callout type="note" %}}
If the model is being used for inference, the extra loss term is eventually automatically discarded, since it is not used. 
So you can just leave it there as part of the model and it will behave correctly automatically.
{{% /callout %}}
