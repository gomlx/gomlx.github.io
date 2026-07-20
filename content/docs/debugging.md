---
title: "Debugging"
section: "Guides"
weight: 30
source: "https://github.com/gomlx/gomlx/blob/main/docs/debugging.md"
---

<img align="right" src="/img/gopher_debugging.jpeg" alt="GoMLX Gopher Debugging" width="220px"/>

Unfortunately, the computers just "don't get it", and they do what we (or the AI) tell them to do,
and not what we wanted them to do, and programs fail or crash. 
GoMLX provides different ways to track down various types of errors. The most commonly used below:

## Good old "printf"

It's convenient and because of Go fast compilation, often a valid way of developing by just logging
results to the stdout. During graph building development, often one prints the shape of the `Node`
being operated to confirm (or not) one's expectations.

## Execution Logging

Every `Node` in the computation graph can be marked for logging during the graph construction phase. This is done by calling `Node.SetLogged(msg)` or `Node.SetLoggedf(format, args...)` on the node of interest.

At execution time, the executor (`Exec`) automatically:
1. Detects all nodes in the graph that have been marked for logging.
2. Appends these nodes as extra outputs to the underlying graph execution, ensuring their tensor values are computed.
3. Retrieves the computed tensor values post-execution and passes the messages, tensor values, and node IDs to the registered `LoggerFn` callback.
4. Slices out the logged nodes' values from the final output list returned to the user, making execution logging completely transparent to the caller of `Exec.Call` (or `MustCall`).

By default, the executor is configured with `DefaultNodeLogger`, which prints the graph's name, the node ID, the message, and the computed value of the tensor to stdout. 
* If the message begins with the prefix `"#full "`, the default logger will output the full tensor value using `Tensor.GoStr()`. Otherwise, a truncated preview of the tensor is printed.

You can customize this behavior or implement your own logger by calling `Exec.SetNodeLogger(loggerFn)`. The signature for the callback is:

```go
type LoggerFn func(graph *Graph, messages []string, values []*tensors.Tensor, nodes []NodeId)
```

To disable execution logging entirely, you can call `exec.SetNodeLogger(nil)`.

### Example

<!-- sync_code: file=debugging/logging/main.go tag=logging -->
```go
func SquareAndAddOne(x *Node) *Node {
	xSquared := Mul(x, x)
	// Mark the intermediate square calculation to be logged:
	xSquared.SetLoggedf("#full x^2 intermediate value")

	one := Scalar(x.Graph(), x.DType(), 1.0)
	return Add(xSquared, one)
}

exec := MustNewExec(backend, SquareAndAddOne)

// Option 1: Run with the default logger (prints to stdout)
exec.MustCall([]float32{2.0, 3.0})

// Option 2: Run with a custom logger
exec.SetNodeLogger(func(g *Graph, messages []string, values []*tensors.Tensor, nodes []NodeId) {
	for i, msg := range messages {
		fmt.Printf("Custom Logger: [Node #%d] %s = %v\n", nodes[i], msg, values[i].Value())
	}
})
exec.MustCall([]float32{5.0, 7.0})
```
<div align="right"><small><a href="https://github.com/gomlx/gomlx/blob/main/examples/gomlx.github.io/debugging/logging/main.go#L16">(See source)</a></small></div>

Output:

<!-- sync_code: file=debugging/logging/main.go output_tag=logging -->
```
DefaultNodeLogger(Graph "Exec:main.SquareAndAddOne#1"):
	(Node #1) x^2 intermediate value: (Float32)[2]: []float32{4, 9}
Custom Logger: [Node #1] #full x^2 intermediate value = [25 49]
```


## NanLogger

The [`nanlogger.NanLogger`](https://pkg.go.dev/github.com/gomlx/gomlx/core/graph/nanlogger) uses GoMLX's logging infrastructure to monitor for `NaN` (Not-a-Number) and `±Inf` (infinity) values during execution or training. 

Debugging numerical instability in deep learning model graphs can be extremely challenging, as `NaN`s propagate rapidly across operations. `NanLogger` solves this by tracing specific nodes and identifying the *first* node (closest to inputs) where a `NaN`/`Inf` is detected, capturing its call stack trace and user-defined scope.

Under the hood, tracing a node adds a finite check node to the graph (using `graph.IsFinite(node)` and `graph.LogicalAll`) marked with `SetLogged(uniqueMessageID)`. The `NanLogger` registers itself as the executor's logger by calling `AttachToExec(exec)` or `AttachToTrainer(trainer)`. It acts as a pass-through logger: any log messages that do not belong to `NanLogger` are forwarded to the executor's previous logger.

#### Handlers

When a `NaN` or `±Inf` is encountered, it triggers a `HandlerFn` to report the issue. `NanLogger` provides several built-in handlers:
* **`ReportScopeHandler` (default)**: Logs the active scope path of the failure using `klog.Infof`.
* **`ReportAllHandler`**: Logs the active scope path and the full Go call stack trace where `Trace` was set.
* **`ReportAndPanicHandler`**: Logs the active scope and stack trace, and then panics. This is highly useful during active debugging to halt execution immediately.
* **`DefaultBreakHandler`**: Logs the details using `klog.Errorf`.

You can set the desired handler using `WithHandler(handlerFn)`.

#### Managing Scope and Traces
* **Scopes**: You can push and pop layers/operation names onto the logger's scope stack using `PushScope("layer_name")` and `PopScope()`. Any trace created will inherit the current scope hierarchy.
* **First NaN vs. Continue**:
  * `TraceFirstNaN(node, scope...)`: Traces only the first NaN occurrence. If multiple traced nodes are non-finite, only the one closest to the input is handled.
  * `TraceAndContinue(node, scope...)`: Logs the non-finite occurrence but does not stop execution.
  * `Trace(node, scope...)`: Behaves according to the `WithStopAtFirst(bool)` configuration (default is true).

### Example

<!-- sync_code: file=debugging/nanlogger/main.go tag=nanlogger -->
```go
func MyLayer(l *nanlogger.NanLogger, x *Node) *Node {
	l.PushScope("dense_layer_1")
	defer l.PopScope()

	denominator := Scalar(x.Graph(), x.DType(), 0.0)
	y := Div(x, denominator)
	l.TraceFirstNaN(y, "output_division")
	return y
}

l := nanlogger.New().WithHandler(nanlogger.ReportAndPanicHandler)
exec := MustNewExec(backend, func(x *Node) *Node {
	return MyLayer(l, x)
})
l.AttachToExec(exec)
l.WithHandler(nanlogger.ReportScopeHandler)
_, err := exec.Call([]float32{1.0, 2.0})
```
<div align="right"><small><a href="https://github.com/gomlx/gomlx/blob/main/examples/gomlx.github.io/debugging/nanlogger/main.go#L16">(See source)</a></small></div>

Output:

<!-- sync_code: file=debugging/nanlogger/main.go output_tag=nanlogger -->
```
I0720 18:14:08.141136  135770 nanlogger.go:375] NaN/Inf observed for scope ["dense_layer_1" "output_division"]
```


## Node with Stack-Traces

When writing gradient of new operations (or any other debugging) sometimes it's useful to know
exactly where a `Node` was created. Use `Graph.SetTraced(true)` to enable all new nodes to include
its stack-trace. And `Node.Trace()` to access it for printing or debugging.
