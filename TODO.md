# dnn.cool

Some dumb stuff I am playing with.

DOING:

- [ ] Per-task result interpretation
- [ ] Overall interpretation
- [ ] Cleaner loss function reducing - separate methods for reduced and non-reduced loss
- [ ] Publish best/worst images to Tensorboard
- [ ] Grad-CAM per branch

TODO:

- [ ] Implement some common decoders
- [ ] Implement some common metrics
- [ ] Implement inference when we don't have ground truth
- [ ] Improve test coverage, especially the user-friendliness part
- [ ] Per-task evaluation information, given that precondition is working correctly.
- [ ] Overall evaluation information
- [ ] Freeze-all but task feature (including Batch Norm) - may include parameter group
- [ ] Set learning rate per task feature
- [ ] Interpretation callback
- [ ] Handles missing labels correctly.
- [ ] Automatic per-task or overall thresholds tuning.
- [ ] UI splitting helper
- [ ] ONNX converter (if possible) - with static analysis.
- [ ] Test training a model with synthetic dataset.
- [ ] Cleaner representations for results
- [ ] Work on error messages
- [ ] Predict only for those, for which precondition is correct.
- [ ] Allow custom entries in the FlowDict
- [ ] Add to latex option (automatically generate loss functions in latex)
- [ ] Implement YOLO for example
- [ ] Add possibility to add weights based on masks
- [ ] Correct operations override for magic methods
- [ ] Improve variable names
- [ ] Add option to skip flatten (for inference it's actually better to keep it, but for loaders it has to be flattened)
- [ ] Customization - add the possibility to pass not only the logits, but other keys as well (maybe include them by default?)
- [ ] Add option for readable output
- [ ] Add a lot of predefined tasks and flows
- [ ] Rethink reduction and variable names
- [ ] Compute only when precondition is True (will require precomputing)
- [ ] Support multilabel classification problem
- [ ] Make sure immutability of objects is preserved
- [ ] Documentation
- [ ] Think of useful methods
- [ ] Think of a better way to organize labels
- [ ] Add the possibility to predict given the precondition is true (very useful for evaluation)
- [ ] Test with very weird cases
- [ ] Receptive field slicing
- [ ] Proper handling of multimetrics (not adding them one by one).
- [ ] Augmentations helper
- [ ] Incremental re-run
- [ ] Support for classification special case handling.
- [ ] Add and implement nice default metrics for different tasks.
- [ ] Think of default values for different tasks.
- [ ] Add per sample activation
- [ ] Spend time thinking about user-friendliness of each class (think about what mistakes would people do?).
- [ ] Good tests about possible gt leak
- [ ] Implement tracing for static analysis (later)
- [ ] Create real dataset by pseudo labeling
- [ ] Rethink user-friendliness API
- [ ] Revisit type hints
- [ ] Auto guess task from column
- [ ] Add the tasks from MMF
- [ ] Add object detection tasks
- [ ] Add NLP tasks
- [ ] Implement smart type guessingusd 
- [ ] Are metrics computed only for when the preconditions are correct?
- [ ] EfficientNet-based project helper
- [ ] Implement smart values converter
- [ ] Implement smart task converter
- [ ] Handle different types when used as precondition, and as ground truth
- [ ] Think of common ways users can shoot themselves in the foot, and add helper messages
- [ ] Make sure multiple inputs are correctly handled.
- [ ] Test nested FC layers
- [ ] Readable/visual result implemenetation - draw bounding boxes, etc.
- [ ] Add precondition builder for tasks other than binary classification task.
- [ ] Add default supervised runner.

Done: 

- [x] Create a helper class for creating tasks (by trying to guess task type) - e.g ProjectHelper.
- [x] Implement Inputs reader.
- [x] Generate a `nn.Module` with `train` mode, which works correctly.
- [x] Generate a `nn.Module` with `eval` mode, which works correctly.
- [x] Per-task loss function `nn.Module`s
- [x] Overall loss function as `nn.Module`
- [x] Per-task metrics
- [x] Overall metrics
- [x] Predictions per task
- [x] Debug why metrics are wrong
- [x] Refactor decorators
- [x] Performance issues
- [x] Think how suitable it would be to use scoping
- [x] Fix bug with the results seem weird
- [x] Optimize inputs (so that it is not read multiple times)
- [x] Make good tests for cascading preconditions (especially for datasets).
- [x] Pass around kwargs for flexibility
- [x] Nested loss functions check and test
- [x] Correct handling when multiple precondition masks are present
- [x] Think of way to handle properly the or operator.
- [x] Can debug the flow in any mode
- [x] Help with logic when creating `nn.Dataset`.
- [x] Callbacks per task (for metrics, loss function, additional metrics, samples, etc.)
- [x] Concept of per-task activation/decoding.
- [x] Overall activation/decoding (topological sorting, etc.).
- [x] Use Python decorators as registers for flow.
- [x] Composite flows (e.g YOLO inside a classifier, etc.)
- [x] Create convincing synthetic dataset
- [x] When using a project, store tasks in flows only when they are needed (will require tracing)
- [x] Fix bug with precondition accumulation when on the same flow
- [x] Add test for predicting on an example and showing decoded
- [x] Treelib explanation
- [x] Test visualizations
- [x] Better handling of `get_inputs`, `get_labels` and `get_dataset`
- [x] Sample decoded results per task
- [x] Use train/test split in tests
- [x] Add assertions for training tests


