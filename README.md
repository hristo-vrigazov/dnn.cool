## DNN Cool: Deep Neural Networks for Conditional objective oriented learning

A bunch of utilities for multi-task learning, not much for now. To get a better idea of the `dnn_cool`'s goals,
you can:
 
* Read a [story](#example-story) about an example application
* Check out its [features](#features)
* Have a look at [code](#code-examples) examples

### Example story

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