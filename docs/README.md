# Modlation Classification

[![GitHub][github-badge]][github]
[![Kaggle][kaggle-badge]][kaggle]
[![Binder][binder-badge]][binder]
[![License][license-badge]][license]

- [Modlation Classification](#modlation-classification)
  - [The Dataset](#the-dataset)
  - [The Models](#the-models)
    - [Hyperparameters](#hyperparameters)
    - [CNN model](#cnn-model)
      - [Experiment 1](#experiment-1)
      - [Experiment 2](#experiment-2)
      - [Experiment 3](#experiment-3)
      - [Experiment 4](#experiment-4)

## The Dataset

<!-- TODO: Talk about the dataset -->

## The Models

### Hyperparameters

| Hyperparameter | Value      |
| -------------- | ---------- |
| Early stopping | patience=5 |
| Optimizer      | Adam       |
| Max epochs     | 100        |

### CNN model

![CNN model][cnn-model]

```text
Model: "cnn_2_ch"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 2, 128, 64)        640
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 1, 64, 16)         9232
_________________________________________________________________
flatten (Flatten)            (None, 1024)              0
_________________________________________________________________
dense (Dense)                (None, 128)               131200
_________________________________________________________________
dense_1 (Dense)              (None, 11)                1419
=================================================================
Total params: 142,491
Trainable params: 142,491
Non-trainable params: 0
_________________________________________________________________
```

#### Experiment 1

Learning rate: 0.001

| Parameter           | Value  |
| ------------------- | ------ |
| Training loss       | 1.0292 |
| Training accuracy   | 0.5698 |
| Validation loss     | 1.0691 |
| Validation accuracy | 0.5485 |
| Number of epochs    | 20     |

![Experiment 1](img/exp-1-loss.png)
![Experiment 1](img/exp-1-acc.png)

#### Experiment 2

Learning rate: 0.003

| Parameter           | Value  |
| ------------------- | ------ |
| Training loss       | 1.1237 |
| Training accuracy   | 0.5308 |
| Validation loss     | 1.1264 |
| Validation accuracy | 0.5321 |
| Number of epochs    | 18     |

![Experiment 2](img/exp-2-loss.png)
![Experiment 2](img/exp-2-acc.png)

#### Experiment 3

- Learning rate scheduler: step decay
- Learning rate start: 0.003

| Parameter           | Value  |
| ------------------- | ------ |
| Training loss       | 1.0804 |
| Training accuracy   | 0.5449 |
| Validation loss     | 1.0917 |
| Validation accuracy | 0.5388 |
| Number of epochs    | 24     |


![Experiment 3](img/exp-3-loss.png)
![Experiment 3](img/exp-3-acc.png)

#### Experiment 4

- Learning rate scheduler: step decay
- Learning rate start: 0.0009

| Parameter           | Value  |
| ------------------- | ------ |
| Training loss       | 1.0417 |
| Training accuracy   | 0.5631 |
| Validation loss     | 1.0663 |
| Validation accuracy | 0.5519 |
| Number of epochs    | 24     |

![Experiment 4](img/exp-4-loss.png)
![Experiment 4](img/exp-4-acc.png)


### RNN model

```text
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
simple_rnn_2 (SimpleRNN)     (None, 2, 128)            32896     
_________________________________________________________________
simple_rnn_3 (SimpleRNN)     (None, 64)                12352     
_________________________________________________________________
dense_1 (Dense)              (None, 10)                650       
=================================================================
Total params: 45,898
Trainable params: 45,898
Non-trainable params: 0
_________________________________________________________________
```

#### Experiment 1

Learning rate: 0.001

| Parameter           | Value  |
| ------------------- | ------ |
| Training loss       | 1.1983 |
| Training accuracy   | 0.4975 |
| Validation loss     | 1.2184 |
| Validation accuracy | 0.4925 |
| Number of epochs    | 25     |

![Experiment1 RNN](img/exp1-rnn-acc.png)

#### Experiment 2

Learning rate: 0.0001

| Parameter           | Value  |
| ------------------- | ------ |
| Training loss       | 1.4268 |
| Training accuracy   | 0.4152 |
| Validation loss     | 1.4482 |
| Validation accuracy | 0.4085 |
| Number of epochs    | 30     |

![Experiment2 RNN](img/exp2-rnn-acc.png)

#### Experiment 3

- Learning rate scheduler: step decay
- Learning rate start: 0.001

| Parameter           | Value  |
| ------------------- | ------ |
| Training loss       | 1.2141 |
| Training accuracy   | 0.4769 |
| Validation loss     | 1.2154 |
| Validation accuracy | 0.4698 |
| Number of epochs    | 35     |


![Experiment3 RNN](img/exp3-rnn-acc.png)

#### Experiment 4

- Learning rate scheduler: step decay
- Learning rate start: 0.0001

| Parameter           | Value  |
| ------------------- | ------ |
| Training loss       | 1.3328 |
| Training accuracy   | 0.4421 |
| Validation loss     | 1.3417 |
| Validation accuracy | 0.4375 |
| Number of epochs    | 100    |

![Experiment4 RNN](img/exp4-rnn-acc.png)

#### Experiment 5

- Learning rate scheduler: step decay
- Learning rate start: 0.0009

| Parameter           | Value  |
| ------------------- | ------ |
| Training loss       | 1.0742 |
| Training accuracy   | 0.5445 |
| Validation loss     | 1.1397 |
| Validation accuracy | 0.5245 |
| Number of epochs    | 63     |

![Experiment5 RNN](img/exp5-rnn-acc.png)
![Experiment5 RNN](img/exp5-rnn-SNRRacc.png)


#### With First Derivative

- Learning rate scheduler: step decay
- Learning rate start: 0.0008

| Parameter           | Value  |
| ------------------- | ------ |
| Training loss       | 1.3695 |
| Training accuracy   | 0.4134 |
| Validation loss     | 1.4149 |
| Validation accuracy | 0.5245 |
| Test accuracy       | 0.4013 |
| Number of epochs    | 55     |

![RNN Derivative](img/rnn-der-acc.png)
![RNN Derivative](img/rnn-der-SNRRacc.png)





<!-- References -->
[github]: github.com/moharamfatema/modulation-classification
[github-badge]: https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white

[license]: https://github.com/moharamfatema/modulation-classification/blob/main/COPYING

[license-badge]: https://img.shields.io/badge/License-GPLv3-blue?style=for-the-badge

[kaggle]: kaggle.com/fatemamoharam/modulation-classification

[kaggle-badge]: https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white

[binder-badge]: https://img.shields.io/badge/launch-binder-orange.svg?style=for-the-badge

[binder]: https://mybinder.org/v2/gh/moharamfatema/modulation-classification/main

[cnn-model]: https://github.com/moharamfatema/modulation-classification/blob/main/docs/img/cnn-model.png

