# dnn.cool

Task:


- [x] Generate a `nn.Module` with `train` mode, which work correctly.
- [x] Generate a `nn.Module` with `eval` mode, which work correctly.
- [ ] Per-task evaluation information, given that precondition is working correctly.
- [ ] Overall evaluation information
- [ ] Per-task result interpretation
- [ ] Overall interpretation
- [ ] Per-task loss function `nn.Module`s
- [ ] Overall loss function as `nn.Module`
- [ ] Per-task metrics
- [ ] Overall metrics
- [ ] Predictions per task
- [ ] Freeze-all but task feature (including Batch Norm) - may include parameter group
- [ ] Set learning rate per task feature
- [ ] Callbacks per task (for metrics, loss function, additional metrics, samples, etc.)
- [ ] Sample decoded results per task
- [ ] Handles missing labels correctly.
- [ ] Concept of per-task activation/decoding.
- [ ] Overall activation/decoding (topological sorting, etc.).
- [ ] Automatic per-task or overall hyperparameter tuning.
- [ ] Composite tasks (e.g YOLO)
- [ ] UI splitting helper
- [ ] ONNX converter (if possible) - with static analysis.
- [ ] Test training a model with synthetic dataset.
- [ ] Help with logic when creating `nn.Dataset`.
- [ ] Treelib explanation
- [ ] Grad-CAM per branch
- [ ] Can debug the flow in any mode
- [ ] Cleaner representations for results
- [ ] Work on error messages
- [ ] Predict only for those, for which precondition is correct.
- [ ] Allow custom entries in the FlowDict
- [ ] Add to latex option (automatically generate loss functions in latex)
- [ ] Think of way to handle properly the or operator.
