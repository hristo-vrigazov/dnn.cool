## DNN Cool: Deep Neural Networks for Conditional objective oriented learning

A bunch of utilities for multi-task learning, not much for now. To get a better idea of the `dnn_cool`'s goals,
you can:
 
* Read a [story](#motivational-story) about an example application
* Check out its [features](#features)
* Have a look at [code](#code-examples) examples
* Interested in [how it works](#how-does-it-work)?

Installation is as usual:

```bash
pip install dnn_cool
```

### Motivational story

Let's say that you are building a home security system. You have collected a bunch of images, and have annotated them. 
So you have a few columns:

* `camera_blocked` - whether the camera is blocked
* `door_open` - whether the door of the apartment is open (has meaning only when the camera is not blocked)
* `door_locked` - whether the door of the apartment is locked (has meaning only when the door is closed)
* `person_present` - whether a person is present (has meaning only when the door is opened)
* `face_x1`, `face_y1`, `face_w`, `face_h` - the coordinates of the face of the person (has meaning only when a person 
is present)
* `body_x1`, `body_y1`, `body_w`, `body_h` - the coordinates of the body the person (has meaning only when a person is 
present)
* `facial_characteristics` - represents characteristics of the face, like color, etc. (has meaning only when a person 
is present)
* `shirt_type` - represents a specific type of shirt (e.g white shirt, etc.)

Let's say that you also could not annotate all of the images, so we will put `nan` when there is no annotation.

Here's an example of the dataframe with the data you have obtained:

|    |   camera_blocked |   door_open |   person_present |   door_locked |   face_x1 |   face_y1 |   face_w |   face_h | facial_characteristics   |   body_x1 |   body_y1 |   body_w |   body_h |   shirt_type | img    |
|---:|-----------------:|------------:|-----------------:|--------------:|----------:|----------:|---------:|---------:|:-------------------------|----------:|----------:|---------:|---------:|-------------:|:-------|
|  0 |              nan |           0 |                0 |             1 |       nan |       nan |      nan |      nan | nan                      |       nan |       nan |      nan |      nan |          nan | 0.jpg  |
|  1 |              nan |           1 |                1 |           nan |        22 |         4 |       12 |       12 | 0,1                      |        22 |        15 |       12 |       29 |            2 | 1.jpg  |
|  2 |              nan |           0 |                0 |             0 |       nan |       nan |      nan |      nan | nan                      |       nan |       nan |      nan |      nan |          nan | 2.jpg  |
|  3 |              nan |           1 |                1 |           nan |        20 |         3 |       12 |       12 | 0,1                      |        21 |        15 |       10 |       29 |            1 | 3.jpg  |
|  4 |              nan |           1 |                0 |           nan |       nan |       nan |      nan |      nan | nan                      |       nan |       nan |      nan |      nan |          nan | 4.jpg  |
|  5 |              nan |           1 |                1 |           nan |        22 |         2 |       12 |       12 | 0,1                      |        22 |        13 |       11 |       30 |            2 | 5.jpg  |
|  6 |                1 |         nan |              nan |           nan |       nan |       nan |      nan |      nan | nan                      |       nan |       nan |      nan |      nan |          nan | 6.jpg  |
|  7 |                0 |           1 |                1 |           nan |        24 |         2 |       12 |       12 | 2                        |        25 |        12 |       13 |       31 |            2 | 7.jpg  |
|  8 |                0 |           1 |                0 |           nan |       nan |       nan |      nan |      nan | nan                      |       nan |       nan |      nan |      nan |          nan | 8.jpg  |
|  9 |                0 |           1 |                1 |           nan |        20 |         0 |       12 |       12 | 0,2                      |        20 |        10 |       10 |       28 |            4 | 9.jpg  |
| 10 |                0 |           1 |                1 |           nan |        24 |         0 |       12 |       12 | 0                        |        25 |        13 |       11 |       31 |            3 | 10.jpg |
| 11 |                0 |           1 |                1 |           nan |        23 |         6 |       12 |       12 | 0,1                      |        23 |        19 |       11 |       31 |            1 | 11.jpg |
| 12 |                0 |           0 |                0 |             0 |       nan |       nan |      nan |      nan | nan                      |       nan |       nan |      nan |      nan |          nan | 12.jpg |
| 13 |                0 |           1 |                1 |           nan |        22 |         1 |       12 |       12 | 2                        |        20 |        11 |       13 |       29 |            2 | 13.jpg |
| 14 |                1 |         nan |              nan |           nan |       nan |       nan |      nan |      nan | nan                      |       nan |       nan |      nan |      nan |          nan | 14.jpg |
| 15 |                0 |           1 |                0 |           nan |       nan |       nan |      nan |      nan | nan                      |       nan |       nan |      nan |      nan |          nan | 15.jpg |
| 16 |                0 |           1 |                1 |           nan |        22 |         0 |       12 |       12 | 0                        |        22 |        10 |       11 |       28 |            1 | 16.jpg |
| 17 |                0 |           1 |                0 |           nan |       nan |       nan |      nan |      nan | nan                      |       nan |       nan |      nan |      nan |          nan | 17.jpg |
| 18 |                0 |           1 |                1 |           nan |        24 |         1 |       12 |       12 |                          |        22 |        11 |       13 |       28 |            2 | 18.jpg |
| 19 |                0 |           0 |                0 |             0 |       nan |       nan |      nan |      nan | nan                      |       nan |       nan |      nan |      nan |          nan | 19.jpg |

Now, how would you approach this? Let's summarize the options below

1. Train a neural network for every small task - this will definitely work, but will be extremely heavy (you will have 
14 different neural networks) - moreover, every model will extract similar low-level features. Will have huge
memory and computational requirements, will end up very expensive.
2. Use a pretrained face detector/body detector and train neural network per every other task - this suffers from the same
issues before, there are still going to be a lot of neural networks, plus you don't know if the pretrained detector
would be suitable for your use case, since detectors usually are pretrained for a large number of classes. Also, notice
how after we have detected the face, we also want to predict the facial characteristics, and for the body we want to 
predict the shirt type - this means that we would have to train these separately, or we would have to understand
the detector in detail to know how to modify it.
3. Multi-task learning approach, where some initial features are extracted, then from a certain point on,
there are branches for every task separately. This is the most lightweight and the best option in terms of quality 
(because it is a well known fact that multi-task learning improves generalization)
but it is not so trivial to implement. Notice that for example when the camera is blocked, we should not backpropagate
for the other tasks, since there is no way for us to know where the person would be etc. The model has to learn **conditional 
objectives**, which are not so easy to keep track of for several reasons.
 
This library `dnn_cool` aims to make the conditional objectives approach much easier, therefore is named **D**eep 
**N**eural **N**etworks for **C**onditional **O**bjectives **o**riented **l**earning.

Assume that the variable `df` holds the annotated data. Then we would have to create a `dnn_cool.project.Project`,
where first we have to specify which columns should be treated as output tasks, and what type of values each column
is.

```python
output_col = ['camera_blocked', 'door_open', 'person_present', 'door_locked',
              'face_x1', 'face_y1', 'face_w', 'face_h',
              'body_x1', 'body_y1', 'body_w', 'body_h']

converters = Converters()

type_guesser = converters.type
type_guesser.type_mapping['camera_blocked'] = 'binary'
type_guesser.type_mapping['door_open'] = 'binary'
type_guesser.type_mapping['person_present'] = 'binary'
type_guesser.type_mapping['door_locked'] = 'binary'
type_guesser.type_mapping['face_x1'] = 'continuous'
type_guesser.type_mapping['face_y1'] = 'continuous'
type_guesser.type_mapping['face_w'] = 'continuous'
type_guesser.type_mapping['face_h'] = 'continuous'
type_guesser.type_mapping['body_x1'] = 'continuous'
type_guesser.type_mapping['body_y1'] = 'continuous'
type_guesser.type_mapping['body_w'] = 'continuous'
type_guesser.type_mapping['body_h'] = 'continuous'
type_guesser.type_mapping['img'] = 'img'
```

Then we have to specify how would we convert the values from a given dataframe column to actual tensor values.
For example, the column with the image filename is converted into a tensor by reading the filename from disk, binary 
values are just converted to bool tensor (with missing values marked), and continous variables are normalized
in `[0, 1]`. Note that we can also specify a converter directly for a given column (skipping its type), by using
`values_converter.col_mapping`, but we will describe this in more detail later.

```python
values_converter = converters.values
values_converter.type_mapping['img'] = imgs_from_disk_converter
values_converter.type_mapping['binary'] = binary_value_converter
values_converter.type_mapping['continuous'] = bounded_regression_converter

task_converter = converters.task
```

Now, we tell how a given type of values is converted to a `Task`. A `Task` is a collection of a Pytorch `nn.Module`,
which holds the weights specific for the task, the activation of the task, the decoder of the task, and the loss function
of the task.

```python
task_converter.type_mapping['binary'] = binary_classification_task
task_converter.type_mapping['continuous'] = bounded_regression_task

project = Project(df,
                  input_col='img',
                  output_col=output_col,
                  converters=converters,
                  project_dir='./high-level-project')
```

Perfect! Now it's time to start to start adding `TaskFlow`s - this is basically a function that describes
the dependencies between the tasks. `TaskFlow` actually extends `Task`, so you can use `TaskFlow` inside a `TaskFlow`, 
etc. Let's first start with creating a `TaskFlow` for the face-related tasks:

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

Here, `flow` is the object which holds all tasks so far, `x` is a `Dict` which holds features for the 
different branches, and `out` is the variable in which the final result is accummulated.

Now we can add a `TaskFlow` for body-related tasks as well:

```python
@project.add_flow
def body_regression(flow, x, out):
    out += flow.body_x1(x.body_localization)
    out += flow.body_y1(x.body_localization)
    out += flow.body_w(x.body_localization)
    out += flow.body_h(x.body_localization)
    out += flow.shirt_type(x.features)
    return out
``` 

Since these two flows are already added, let's group them into a `TaskFlow` for person-related tasks:

```python
@project.add_flow
def person_regression(flow, x, out):
    out += flow.face_regression(x)
    out += flow.body_regression(x)
    return out
```

And now let's implement the full flow of tasks:

```python
@project.add_flow
def full_flow(flow, x, out):
    out += flow.camera_blocked(x.features)
    out += flow.door_open(x.features) | (~out.camera_blocked)
    out += flow.door_locked(x.features) | (~out.door_open)
    out += flow.person_present(x.features) | out.door_open
    out += flow.person_regression(x) | out.person_present
    return out
```

Here you can notice the `|` operator, which is used as a precondition (read it as `given`, e.g 
"Door open given that the camera is not blocked"). Let's now get the full flow for the project
and have a look at some of its methods.

```python
flow = project.get_full_flow()
flow.get_loss()                 # returns a loss function that uses the children's loss functions
flow.get_per_sample_loss()      # returns a loss function that will return a loss item for every sample; useful for interpreting results
flow.torch()                    # returns a `nn.Module` that uses the children's modules, according to the logic described in the task flow
flow.get_dataset()              # returns a Pytorch `Dataset` class, which includes the preconditions needed to know which weights should be updated
flow.get_metrics()              # returns a list of metrics, which consists of the metrics of its children
flow.get_treelib_explainer()    # returns an object that when called, draws a tree of the decision making, based on the task flow
flow.get_decoder()              # returns a decoder, which decodes all children tasks
flow.get_activation()           # returns an activation function, which invokes the activations of its children
flow.get_evaluator()            # returns an evaluator, which evaluates every task (given that its respective precondition is satisfied)
```
 
### Features

Main features are:

* Support for task preconditioning - you can say that a task is a precondition to another task
* Support for nested tasks - tasks which contain other tasks
* Support for handling missing values 
* Treelib explanation generator, for example:

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

### Code examples

### How does it work?