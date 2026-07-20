---
title: "Training Loop"
section: "Training & Layers"
weight: 20
---

<img align="right" src="/img/gopher_training.jpg" alt="GoMLX Gopher" width="220px"/>

In GoMLX, the training process is orchestrated by two primary objects:

1. **`train.Trainer`**: Orchestrates running the model function, calculating the loss, backpropagating gradients through the optimizer, and computing metrics. It represents a single training or evaluation step.
2. **`train.Loop`**: Drives the dataset iterations over multiple training steps, manages execution states, and coordinates periodic tasks (such as saving checkpoints, logging, or plotting metrics) via callbacks.

---

## The Trainer (`train.Trainer`)

The `Trainer` is created via `train.NewTrainer`. It compiles the actual training and evaluation steps into computation graphs. Under the hood, the trainer:
* Prepares the `model.Scope` to create or retrieve variables.
* Builds the graph that calls the model function, the loss function, and the optimizer.
* Handles JIT-compiling of the training step for the specific shapes in your batch.

Here is how you set up a typical `Trainer`:

<!-- sync_code: file=training/main.go tag=trainer_creation -->
```go
backend := compute.MustNew()
store := model.NewStore()

// Configure hyperparameters in the model store
store.SetParams(map[string]any{
	"batch_size":                128,
	"train_steps":               1000,
	optimizer.ParamOptimizer:    "adam",
	optimizer.ParamLearningRate: 0.001,
	activation.ParamActivation:  "relu",
	fnn.ParamNumHiddenLayers:    2,
	fnn.ParamNumHiddenNodes:     64,
})
scope := store.RootScope()
modelFn := func(scope *model.Scope, spec any, inputs []*Node) []*Node {
	// In a real example, inputs would contain calibrated continuous and categorical features.
	// For simplicity, we just run a Feed-Forward Neural Network on the continuous features.
	continuousFeatures := inputs[1]
	logits := fnn.New(scope, continuousFeatures, 1).Done()
	return []*Node{logits}
}

// 4. Configure metrics
// Moving average is recommended for training since the model changes at each step.
movingAccuracy := metric.NewMovingAverageBinaryLogitsAccuracy("Moving Average Accuracy", "~acc", 0.01)
// Mean accuracy is recommended for evaluation because the model is frozen.
meanAccuracy := metric.NewMeanBinaryLogitsAccuracy("Mean Accuracy", "#acc")

// 5. Create the Trainer
trainer := train.NewTrainer(
	backend,
	store,
	modelFn,
	loss.BinaryCrossentropyLogits,
	optimizer.Adam().LearningRate(0.001).Done(),
	[]metric.Interface{movingAccuracy}, // metrics evaluated at train time
	[]metric.Interface{meanAccuracy},   // metrics evaluated at eval time
)
```
<div align="right"><small><a href="https://github.com/gomlx/gomlx/blob/main/examples/gomlx.github.io/training/main.go#L39">(See source)</a></small></div>

### Key Parameters:
* **`modelFn`**: Any function matching the `ModelFnCompatible` constraint. GoMLX accepts multiple function signatures (e.g., omitting the `spec` parameter or returning 1-3 nodes) and wraps them automatically.
* **`lossFn`**: A loss function (like `loss.BinaryCrossentropyLogits` or `loss.MeanSquaredError`). If the loss is not a scalar, GoMLX automatically applies a mean reduction.
* **`optimizer`**: An implementation of `optimizer.Interface` (such as `optimizer.Adam` or `optimizer.SGD`).
* **`trainMetrics`**: A slice of `metric.Interface` computed at training time. **Moving average metrics** (e.g., `metric.NewMovingAverageBinaryLogitsAccuracy`) are recommended here since the model parameters change at every step.
* **`evalMetrics`**: Metrics computed during validation. **Mean metrics** (e.g., `metric.NewMeanBinaryLogitsAccuracy`) are recommended because validation evaluates a frozen model over a dataset exactly once.

---

## The Training Loop (`train.Loop`)

To run the trainer over a dataset, you wrap it in a `train.Loop` object:

<!-- sync_code: file=training/main.go tag=loop_creation -->
```go
loop := train.NewLoop(trainer)
```
<div align="right"><small><a href="https://github.com/gomlx/gomlx/blob/main/examples/gomlx.github.io/training/main.go#L92">(See source)</a></small></div>

The loop maintains the state of execution (like the current step `l.LoopStep`) and exposes a registry to attach callback functions.

---

## Attaching Callbacks (Periodic Tasks)

A key power of `train.Loop` is the ability to attach periodic tasks that trigger at specific steps or time intervals. GoMLX includes standard helpers to attach progress bars, checkpoints, and metrics plotting:

<!-- sync_code: file=training/main.go tag=loop_callbacks -->
```go
// A. Attach a progress bar to monitor progress in the terminal
commandline.ProgressbarStyle = progressbar.ThemeUnicode
commandline.AttachProgressBar(loop)

// B. Setup checkpoint saving (every minute or every N steps)
var checkpointHandler *checkpoint.Handler
if *flagCheckpoint != "" {
	checkpointHandler, _ = checkpoint.Build(store).
		DirFromBase(*flagCheckpoint, *flagDataDir).
		Keep(3).
		Done()

	// Save checkpoints periodically every minute
	train.PeriodicCallback(loop, time.Minute, true, "checkpoint", 100, checkpointHandler.SaveOnStepFn)
}

// C. Attach Plotly to automatically generate training plots
plotly.New().
	WithCheckpoint(checkpointHandler).
	Dynamic().
	WithDatasets(trainEvalDS, testEvalDS).
	ScheduleExponential(loop, 100, 1.2)

// D. Register a custom callback to print moving average metrics
train.EveryNSteps(loop, 500, "log_metrics", 0, func(l *train.Loop, metrics []*tensors.Tensor) error {
	// metrics[0] is the batch loss, metrics[1] is the moving average loss, etc.
	fmt.Printf("\nStep %d: Loss = %.4f, Moving Acc = %.4f\n", l.LoopStep, metrics[0].Value(), metrics[2].Value())
	return nil
})
```
<div align="right"><small><a href="https://github.com/gomlx/gomlx/blob/main/examples/gomlx.github.io/training/main.go#L98">(See source)</a></small></div>

### A. Terminal Progress Bar
`commandline.AttachProgressBar(loop)` attaches a terminal-based progress bar (using the `schollz/progressbar` library) that displays training speed, steps completed, estimated time remaining, and the moving average loss.

### B. Checkpoint Saving
`train.PeriodicCallback` allows you to trigger a callback periodically. In this case, we register the `checkpointHandler.SaveOnStepFn` function to save model weights and hyperparameters every minute (or every N steps).

### C. Live Plotly Plots
Using `plotly.New()`, you can generate dynamic, interactive HTML plots of your metrics. `plotly` hooks into the checkpoint handler to save plots as files, and schedules updates exponentially (e.g., plotting frequently at the start of training and less frequently as training converges).

### D. Custom Callbacks
`train.EveryNSteps` registers a custom callback function. The callback is passed the active `train.Loop` and the evaluated metrics (in a slice of `*tensors.Tensor`), allowing you to print values, adjust learning rates, or execute custom monitoring logic.

---

## Running the Loop

Once the trainer, loop, and callbacks are configured, you start training by passing a dataset iterator:

<!-- sync_code: file=training/main.go tag=loop_execution -->
```go
trainSteps := model.GetParamOr(scope, "train_steps", 1000)
globalStep := int(optimizer.GetGlobalStep(scope))
if globalStep != 0 {
	fmt.Printf("Restarting training from global step %d\n", globalStep)
}

_, err := loop.RunToGlobalStep(trainDS, trainSteps)
if err != nil {
	panic(err)
}
fmt.Println("Training complete!")
```
<div align="right"><small><a href="https://github.com/gomlx/gomlx/blob/main/examples/gomlx.github.io/training/main.go#L130">(See source)</a></small></div>

### Execution Options: `RunSteps`, `RunToGlobalStep`, and `RunEpochs`
* **`loop.RunSteps(dataset, steps)`**: Runs the training loop for an absolute number of steps from the current state.
* **`loop.RunToGlobalStep(dataset, targetStep)`**: Preferred when training can be restarted from checkpoints. It checks the global step stored in the model's parameters and only trains for the remaining steps to reach `targetStep`. If `targetStep` is already reached, it returns immediately.
* **`loop.RunEpochs(dataset, epochs)`**: Runs the loop for a specified number of epochs. An epoch is defined as one complete pass over the dataset. This requires a dataset that is finite (not configured to run infinitely).
