## `dnn_cool`: Deep Neural Networks for Conditional objective oriented learning

To install, just do:

```bash
pip install dnn_cool
```

* [Introduction](#introduction): What is `dnn_cool` in a nutshell?
* [Features](#features): a list of the utilities that `dnn_cool` provides for you
* [Examples](#examples): a simple step-by-step example.
* [Customization](#customization): Learn how to add new tasks, modify them, etc.
* [Inspiration](#inspiration): list of papers and videos which inspired this library

To see the predefined tasks for this release, see [list of predefined tasks](list-of-predefined-tasks)

### Introduction

A framework for multi-task learning, where you may precondition tasks and compose them into bigger tasks.
Many complex neural networks can be trivially implemented with DNN.cool.
For example, creating a neural network that does classification and localization is as simple as:

```python
@project.add_flow
def localize_flow(flow, x, out):
    out += flow.obj_exists(x.features)
    out += flow.obj_x(x.features) | out.obj_exists
    out += flow.obj_y(x.features) | out.obj_exists
    out += flow.obj_w(x.features) | out.obj_exists
    out += flow.obj_h(x.features) | out.obj_exists
    out += flow.obj_class(x.features) | out.obj_exists
    return out
```

If for example you want to classify first if the camera is blocked and then do localization **given that the camera 
is not blocked**, you could do:

```python
@project.add_flow
def full_flow(flow, x, out):
    out += flow.camera_blocked(x.cam_features)
    out += flow.localize_flow(x.localization_features) | (~out.camera_blocked)
    return out
```

Based on these "task flows" as we tell them, `dnn_cool` provides a bunch of [features](#features).
Currently, this is the list of the predefined tasks (they are all located in `dnn_cool.task_flow`):

##### List of predefined tasks

In the current release (0.1.0), the following tasks are availble out of the box:

* `BinaryClassificationTask` - sigmoid activation, thresholding decoder, binary cross entropy loss function. In the 
examples above, `camera_blocked` and `obj_exists` are `BinaryClassificationTask`s.
* `ClassificationTask` - softmax activation, sorting classes decoder, categorical cross entropy loss. In the example 
above, `obj_class` is a `ClassificationTask`
* `MultilabelClassificationTask` - sigmoid activation, thresholding decoder, binary cross entropy loss function.
* `BoundedRegressionTask` - sigmoid activation, rescaling decoder, mean squared error loss function. In the examples 
above, `obj_x`, `obj_y`, `obj_w`, `obj_h` are bounded regression tasks.
* `TaskFlow` - a composite task, that contains a list of children tasks. We saw 2 task flows above. 

### Features

Main features are:

* [Task precondition](#task-preconditioning)
* [Missing values handling](#missing-values)
* [Task composition](#task-composition)
* [Tensorboard metrics logging](#tensorboard-logging)
* [Task interpretations](#task-interpretation)
* [Task evaluation](#task-evaluation)
* [Task threshold tuning](#task-threshold-tuning)
* [Dataset generation](#dataset-generation)
* [Tree explanations](#tree-explanations)

##### Task preconditioning

Use the `|` for task preconditioning (think of `P(A|B)` notation). Preconditioning - ` A | B` means that:

* Include the ground truth for `B` in the input batch when training
* When training, update the weights of the `A` only when `B` is satisfied in the ground truth.
* When training, compute the loss function for `A` only when `B` is satisfied in the ground truth
* When training, compute the metrics for `A` only when `B` is satisfied in the ground truth.
* When tuning threshold for `A`, optimize only on values for which `B` is satisfied in the ground truth.
* When doing inference, compute the metrics for `A` only when the precondition is satisfied according to the decoded
result of the `B` task
* When generating tree explanation in inference mode, do not show the branch for `A` if `B` is not 
satisfied.
* When computing results interpretation, include only loss terms when the precondition is satisfied.

Usually, you have to keep track of all this stuff manually, which makes adding new preconditions very difficult. 
`dnn_cool` makes this stuff easy, so that you can chain a long list of preconditions without worrying you forgot 
something.

##### Missing values

Sometimes for an input you don't have labels for all tasks. With `dnn_cool`, you can just mark the missing label and
`dnn_cool` will update only the weights of the tasks for which labels are available. 

This feature has the awesome property that you don't need a single dataset with all tasks labeled, you can
have different datasets for different tasks and it will work. For example, you can train a single object detection 
neural network that trains its classifier head on ImageNet, and its detection head on COCO.

##### Task composition

You can group tasks in a task flow (we already saw 2 above - `localize_flow` and `full_flow`). You can use this to
organize things better, for example when you want to precondition a whole task flow. For example:

```python
@project.add_flow
def face_regression(flow, x, out):
    out += flow.face_x1(x.face_localization)
    out += flow.face_y1(x.face_localization)
    out += flow.face_w(x.face_localization)
    out += flow.face_h(x.face_localization)
    out += flow.facial_characteristics(x.features)
    return out
```

##### Tensorboard logging

`dnn_cool` logs the metrics per task in Tensorboard, e.g:

![Task loss tensorboard log](./static/task_metric.png)

##### Task interpretation

Also, the best and worst inputs per task are logged in the Tensorboard, for example if the input is an image:

 ![Task interpretation tensorboard log](./static/interpretation_logging.png)

##### Task evaluation
##### Task threshold tuning
##### Dataset generation
##### Tree explanations

```
├── inp 1
│   └── camera_blocked | decoded: [False], activated: [0.], logits: [-117.757324]
│       └── door_open | decoded: [ True], activated: [1.], logits: [41.11258]
│           └── person_present | decoded: [ True], activated: [1.], logits: [60.38873]
│               └── person_regression
│                   ├── body_regression
│                   │   ├── body_h | decoded: [29.672623], activated: [0.46363473], logits: [-0.14571853]
│                   │   ├── body_w | decoded: [12.86382], activated: [0.20099719], logits: [-1.3800735]
│                   │   ├── body_x1 | decoded: [21.34288], activated: [0.3334825], logits: [-0.69247603]
│                   │   ├── body_y1 | decoded: [18.468979], activated: [0.2885778], logits: [-0.9023013]
│                   │   └── shirt_type | decoded: [6 1 0 4 2 5 3], activated: [4.1331367e-23 3.5493638e-17 3.1328378e-26 5.6903808e-30 2.4471377e-25
 2.8071076e-29 1.0000000e+00], logits: [-20.549513  -6.88627  -27.734364 -36.34787  -25.6788   -34.751904
  30.990908]
│                   └── face_regression
│                       ├── face_h | decoded: [11.265154], activated: [0.17601803], logits: [-1.5435623]
│                       ├── face_w | decoded: [12.225838], activated: [0.19102871], logits: [-1.4433397]
│                       ├── face_x1 | decoded: [21.98834], activated: [0.34356782], logits: [-0.64743483]
│                       ├── face_y1 | decoded: [3.2855165], activated: [0.0513362], logits: [-2.9166584]
│                       └── facial_characteristics | decoded: [ True False  True], activated: [9.9999940e-01 1.2074912e-12 9.9999833e-01], logits: [ 14.240071 -27.442476  13.27557 ]

```

but if the model thinks the camera is blocked, then the explanation would be:

```
├── inp 2
│   └── camera_blocked | decoded: [ True], activated: [1.], logits: [76.367676]
```

### Example

### Customization

### Inspiration
