## DNN Cool: Deep Neural Networks for Conditional objective oriented learning

A bunch of utilities for multi-task learning, not much for now. To get a better idea of the `dnn_cool`'s goals,
you can:
 
* Read a [story](#example-story) about an example application
* Check out its [features](#features)
* Have a look at code examples

### Example story


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