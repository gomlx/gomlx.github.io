---
title: "Metrics And Plots"
section: "Training & Layers"
weight: 33
---

Metrics are used to monitor and evaluate a model's performance during both training (on individual batches) and evaluation (over validation or test sets). Plots visualize these metrics over training steps to diagnose issues like underfitting, overfitting, or training instability.

---

## The `metric.Interface`

All metrics in GoMLX implement the `metric.Interface`:

```go
type Interface interface {
	// Name of the metric (e.g., "Sparse Categorical Accuracy").
	Name() string

	// ShortName is a shortened version (e.g. "acc") used in CLI progress bars.
	ShortName() string

	// ScopeName is a unique scope path used to store the metric's temporary state variables.
	ScopeName() string

	// MetricType aggregates metrics sharing the same quantity (e.g., "accuracy", "loss").
	// Metrics of the same type can automatically be plotted together on the same chart axis.
	MetricType() string

	// UpdateGraph builds the computation graph that calculates the metric for the batch.
	// It must return a scalar node.
	UpdateGraph(scope *model.Scope, labels, predictions []*Node) (metric *Node)

	// PrettyPrint converts the metric value tensor to a pretty string.
	PrettyPrint(value *tensors.Tensor) string

	// Reset resets any internal stateful variables (e.g. accumulated weights/counts) at the start of a validation epoch.
	Reset(scope *model.Scope)
}
```

### Training vs. Evaluation Metrics
* **Training Metrics**: Evaluated at every step. Because the model weights are constantly updating, we recommend **Exponential Moving Average** metrics (such as `metric.NewMovingAverageBinaryLogitsAccuracy`).
* **Evaluation Metrics**: Evaluated over a validation/test dataset. Because the model weights are frozen, we recommend **Mean** metrics (such as `metric.NewMeanBinaryLogitsAccuracy`), which accumulate sums and batch counts to calculate a true average over the entire dataset.

---

## Terminal Progress Bar Metrics Table

When you attach the command line progress bar to the training loop:

```go
import "github.com/gomlx/gomlx/ui/commandline"

commandline.ProgressbarStyle = progressbar.ThemeUnicode
commandline.AttachProgressBar(loop)
```

The terminal progress bar displays a continuously updating table of all registered training and evaluation metrics:

```ansi
╭────────────────────────────┬────────────────╮
│                Global Step │ 4_849 of 5_000 │
│ Median train step duration │ 412.3µs        │
│                       Loss │ 0.302          │
│        Moving Average Loss │ 0.276          │
│    Moving Average Accuracy │ 87.17%         │
╰────────────────────────────┴────────────────╯
97%  (1439 steps/s) [3s:0s]
```
---

## Custom Metrics

GoMLX makes it easy to write custom metrics, both stateless (evaluating the last batch only) and stateful (accumulating values over multiple batches).

### 1. Custom Stateless Metric
If your metric does not need to accumulate counters over time, you can create it from any graph function using `metric.NewBaseMetric`:

```go
import (
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/train/metric"
)

// Define the stateless graph function
func CustomAccuracyGraph(_ *model.Scope, labels, predictions []*Node) *Node {
	yTrue := labels[0]
	yPred := predictions[0]
	
	// True if prediction rounded matches labels
	isCorrect := Equal(Round(yPred), yTrue)
	return ReduceMean(ConvertDType(isCorrect, yPred.DType()))
}

// Instantiation:
myStatelessMetric := metric.NewBaseMetric(
	"Custom Accuracy", // Name
	"c_acc",           // ShortName
	"accuracy",        // MetricType
	CustomAccuracyGraph,
	nil,               // Default pretty-printer
)
```

### 2. Custom Stateful Metric
For metrics that require accumulation (like calculating a true average over validation epochs), you must implement the full `metric.Interface`. The metric stores its running sums and sample counts in variables under its own sub-scope (`scope.At(metric.Scope).In(m.ScopeName())`) and resets them inside `Reset(scope)`:

```go
type CustomMeanAccuracy struct {
	// implements metric.Interface
}

func (m *CustomMeanAccuracy) UpdateGraph(scope *model.Scope, labels, predictions []*Node) *Node {
	g := predictions[0].Graph()
	batchAcc := CustomAccuracyGraph(scope, labels, predictions)

	// Access metric variables in the store (marked as non-trainable)
	scope = scope.At(metric.Scope).In(m.ScopeName())
	sumVar := scope.VariableWithValue("sum", 0.0).SetTrainable(false)
	countVar := scope.VariableWithValue("count", 0.0).SetTrainable(false)

	// Update variables
	newSum := Add(sumVar.NodeValue(g), batchAcc)
	newCount := Add(countVar.NodeValue(g), OnesLike(countVar.NodeValue(g)))
	sumVar.SetNodeValue(newSum)
	countVar.SetNodeValue(newCount)

	return Div(newSum, newCount)
}

func (m *CustomMeanAccuracy) Reset(scope *model.Scope) {
	// Reset total sum and count variables in the store to 0
	scope = scope.Store().Scope(path.Join("/", metric.Scope, m.ScopeName()))
	if sumVar := scope.GetVariable("sum"); sumVar != nil {
		sumVar.MustSetValue(tensors.MustFromAnyValue(0.0))
		scope.GetVariable("count").MustSetValue(tensors.MustFromAnyValue(0.0))
	}
}
```

---

## Predefined Metrics

The `github.com/gomlx/gomlx/ml/train/metric` package provides standard classification metrics:

* **Binary Classification**:
  * `NewMeanBinaryAccuracy` / `NewMovingAverageBinaryAccuracy`: Binary accuracy from probabilities.
  * `NewMeanBinaryLogitsAccuracy` / `NewMovingAverageBinaryLogitsAccuracy`: Binary accuracy from raw logits (numerically stable, preferred).
* **Multi-Class Classification (One-Hot)**:
  * `NewMeanCategoricalAccuracy` / `NewMovingAverageCategoricalAccuracy`: Accuracy from categorical probability distributions.
  * `NewMeanCategoricalLogitsAccuracy` / `NewMovingAverageCategoricalLogitsAccuracy`: Accuracy from multi-class raw logits.
* **Multi-Class Classification (Integer Labels)**:
  * `NewMeanSparseCategoricalAccuracy` / `NewMovingAverageSparseCategoricalAccuracy`: Accuracy where labels are integer class indices and predictions are probabilities.
  * `NewMeanSparseCategoricalLogitsAccuracy` / `NewMovingAverageSparseCategoricalLogitsAccuracy`: Accuracy where labels are integer indices and predictions are raw logits.

---

## Training Plots

GoMLX provides plotting utilities in the `github.com/gomlx/gomlx/ui/gonb/plotly` package. Plots are scheduled during training to draw loss and accuracy curves.

```go
import "github.com/gomlx/gomlx/ui/gonb/plotly"

// Inside training configuration:
plotly.New().
	WithCheckpoint(checkpointHandler).
	Dynamic().
	WithDatasets(trainEvalDS, testEvalDS).
	ScheduleExponential(loop, 100, 1.2)
```

### 1. In Jupyter Notebooks (GoNB)
If you train your model inside a Jupyter Notebook using the GoNB kernel, the `.Dynamic()` option tells the library to dynamically render and update interactive, rich Plotly charts directly in the notebook cell outputs.

### 2. Offline / Non-Notebook Environments
In terminal environments (or if a checkpoint handler is configured), Plotly cannot display graphs on screen. Instead:
* It automatically saves the step-by-step metrics data into a JSON file named **`training_plot_points.json`** inside the checkpoint directory.
* **Command-Line Visualization**: You can monitor and render these plots in real-time from another terminal window using the `gomlx_checkpoints` command line tool:
  ```bash
  gomlx_checkpoints -metrics -loop=3s path/to/checkpoint/dir
  ```
* **Plot to HTML (also using Plotly)**:
  ```bash
  gomlx_checkpoints -plot -plot_output=my_model_plots.html path/to/checkpoint/dir
  ```
* **Custom Plotting**: Since `training_plot_points.json` is a standard JSON file containing serialized data points, it can be easily loaded and plotted using any visualization package.
