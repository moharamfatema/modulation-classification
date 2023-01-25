# Modlation Classification

[![GitHub][github-badge]][github]
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
      - [Final Experiment](#final-experiment)
      - [Differentiated Signal CNN](#differentiated-signal-cnn)
      - [Integrated Signal](#integrated-signal)
      - [Combined Features](#combined-features)
      - [Embedded Features (SNR)](#embedded-features-snr)
    - [RNN model](#rnn-model)
      - [Experiment 1](#experiment-1-1)
      - [Experiment 2](#experiment-2-1)
      - [Experiment 3](#experiment-3-1)
      - [Experiment 4](#experiment-4-1)
      - [Experiment 5](#experiment-5)
      - [With First Derivative](#with-first-derivative)
      - [With Integral](#with-integral)
      - [Combination Raw and Derivative](#combination-raw-and-derivative)
      - [Combination Raw and Integral](#combination-raw-and-integral)
      - [Combination Derivative and Integral](#combination-derivative-and-integral)

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

#### Final Experiment

The hyperparameters of this experiment are then used for all CNN models.

The main difference is adding Batch Normalization.

| parameter           | value  |
| ------------------- | ------ |
| Training loss       | 0.9906 |
| Training accuracy   | 0.5909 |
| Validation loss     | 1.0766 |
| Validation accuracy | 0.5575 |
| Number of epochs    | 33     |

![png](modulation_classification_cnn/output_37_2.png)

![png](modulation_classification_cnn/output_37_3.png)

```text
              precision    recall  f1-score   support

        8PSK       0.57      0.54      0.55     36000
      AM-DSB       0.51      0.69      0.59     36000
        BPSK       0.55      0.65      0.59     36000
       CPFSK       0.60      0.64      0.62     36000
        GFSK       0.51      0.69      0.59     36000
        PAM4       0.77      0.70      0.73     36000
       QAM16       0.46      0.32      0.38     36000
       QAM64       0.54      0.45      0.49     36000
        QPSK       0.52      0.55      0.53     36000
        WBFM       0.50      0.31      0.38     36000

    accuracy                           0.55    360000
   macro avg       0.55      0.55      0.55    360000
weighted avg       0.55      0.55      0.55    360000
```

- accuracy at SNR = 0 is around 0.78
- most confused class: WBFM

![png](modulation_classification_cnn/output_41_1.png)

![png](modulation_classification_cnn/output_41_2.png)

![png](modulation_classification_cnn/output_41_3.png)

![png](modulation_classification_cnn/output_41_4.png)

![png](modulation_classification_cnn/output_41_5.png)

![png](modulation_classification_cnn/output_41_6.png)

![png](modulation_classification_cnn/output_41_7.png)

![png](modulation_classification_cnn/output_41_8.png)

![png](modulation_classification_cnn/output_41_9.png)

![png](modulation_classification_cnn/output_41_10.png)

![png](modulation_classification_cnn/output_41_11.png)

![png](modulation_classification_cnn/output_41_12.png)

![png](modulation_classification_cnn/output_41_13.png)

![png](modulation_classification_cnn/output_41_14.png)

![png](modulation_classification_cnn/output_41_15.png)

![png](modulation_classification_cnn/output_41_16.png)

![png](modulation_classification_cnn/output_41_17.png)

![png](modulation_classification_cnn/output_41_18.png)

![png](modulation_classification_cnn/output_41_19.png)

![png](modulation_classification_cnn/output_41_20.png)

![png](modulation_classification_cnn/output_41_21.png)

![png](modulation_classification_cnn/output_41_22.png)

#### Differentiated Signal CNN

| Parameter           | Value  |
| ------------------- | ------ |
| Training loss       | 1.1809 |
| Training accuracy   | 0.5176 |
| Validation loss     | 1.3008 |
| Validation accuracy | 0.4713 |
| Number of epochs    | 15     |

![png](modulation_classification_cnn/output_46_2.png)

![png](modulation_classification_cnn/output_46_3.png)

```text
                  precision    recall  f1-score   support

            8PSK       0.29      0.57      0.38     36000
          AM-DSB       0.44      0.75      0.55     36000
            BPSK       0.60      0.56      0.58     36000
           CPFSK       0.52      0.63      0.57     36000
            GFSK       0.62      0.65      0.63     36000
            PAM4       0.80      0.47      0.59     36000
           QAM16       0.31      0.07      0.12     36000
           QAM64       0.48      0.58      0.53     36000
            QPSK       0.52      0.38      0.44     36000
            WBFM       0.64      0.22      0.33     36000

        accuracy                           0.49    360000
       macro avg       0.52      0.49      0.47    360000
    weighted avg       0.52      0.49      0.47    360000
```

- accuracy at SNR = 0 is around 0.65
- Most confused class: QAM16

![png](modulation_classification_cnn/output_47_1.png)

![png](modulation_classification_cnn/output_47_2.png)

![png](modulation_classification_cnn/output_47_3.png)

![png](modulation_classification_cnn/output_47_4.png)

![png](modulation_classification_cnn/output_47_5.png)

![png](modulation_classification_cnn/output_47_6.png)

![png](modulation_classification_cnn/output_47_7.png)

![png](modulation_classification_cnn/output_47_8.png)

![png](modulation_classification_cnn/output_47_9.png)

![png](modulation_classification_cnn/output_47_10.png)

![png](modulation_classification_cnn/output_47_11.png)

![png](modulation_classification_cnn/output_47_12.png)

![png](modulation_classification_cnn/output_47_13.png)

![png](modulation_classification_cnn/output_47_14.png)

![png](modulation_classification_cnn/output_47_15.png)

![png](modulation_classification_cnn/output_47_16.png)

![png](modulation_classification_cnn/output_47_17.png)

![png](modulation_classification_cnn/output_47_18.png)

![png](modulation_classification_cnn/output_47_19.png)

![png](modulation_classification_cnn/output_47_20.png)

![png](modulation_classification_cnn/output_47_21.png)

![png](modulation_classification_cnn/output_47_22.png)

#### Integrated Signal

| Parameter           | Value  |
| ------------------- | ------ |
| Training loss       | 0.9085 |
| Training accuracy   | 0.6154 |
| Validation loss     | 0.9752 |
| Validation accuracy | 0.5907 |
| Number of epochs    | 50     |

![png](modulation_classification_cnn/output_53_2.png)

![png](modulation_classification_cnn/output_53_3.png)

```text
                  precision    recall  f1-score   support

            8PSK       0.58      0.52      0.55     36000
          AM-DSB       0.47      0.61      0.53     36000
            BPSK       0.77      0.65      0.70     36000
           CPFSK       0.74      0.62      0.68     36000
            GFSK       0.71      0.67      0.69     36000
            PAM4       0.80      0.73      0.76     36000
           QAM16       0.41      0.53      0.46     36000
           QAM64       0.47      0.65      0.55     36000
            QPSK       0.65      0.53      0.58     36000
            WBFM       0.54      0.43      0.48     36000

        accuracy                           0.59    360000
       macro avg       0.61      0.59      0.60    360000
    weighted avg       0.61      0.59      0.60    360000

```

- accuracy at SNR = 0 is around 0.77
- Most confused class: QAM16

![png](modulation_classification_cnn/output_54_1.png)

![png](modulation_classification_cnn/output_54_2.png)

Accuracy at SNR = -10 is 0.4000000059604645 %

![png](modulation_classification_cnn/output_54_4.png)

Accuracy at SNR = -12 is 0.28999999165534973 %

![png](modulation_classification_cnn/output_54_6.png)

Accuracy at SNR = -14 is 0.2199999988079071 %

![png](modulation_classification_cnn/output_54_8.png)

Accuracy at SNR = -16 is 0.15000000596046448 %

![png](modulation_classification_cnn/output_54_10.png)

Accuracy at SNR = -18 is 0.11999999731779099 %

![png](modulation_classification_cnn/output_54_12.png)

Accuracy at SNR = -2 is 0.7300000190734863 %

![png](modulation_classification_cnn/output_54_14.png)

Accuracy at SNR = -20 is 0.10999999940395355 %

![png](modulation_classification_cnn/output_54_16.png)

Accuracy at SNR = -4 is 0.6899999976158142 %

![png](modulation_classification_cnn/output_54_18.png)

Accuracy at SNR = -6 is 0.6100000143051147 %

![png](modulation_classification_cnn/output_54_20.png)

Accuracy at SNR = -8 is 0.5099999904632568 %

![png](modulation_classification_cnn/output_54_22.png)

Accuracy at SNR = 0 is 0.7699999809265137 %

![png](modulation_classification_cnn/output_54_24.png)

Accuracy at SNR = 10 is 0.8199999928474426 %

![png](modulation_classification_cnn/output_54_26.png)

Accuracy at SNR = 12 is 0.8199999928474426 %

![png](modulation_classification_cnn/output_54_28.png)

Accuracy at SNR = 14 is 0.8199999928474426 %

![png](modulation_classification_cnn/output_54_30.png)

Accuracy at SNR = 16 is 0.8199999928474426 %

![png](modulation_classification_cnn/output_54_32.png)

Accuracy at SNR = 18 is 0.8100000023841858 %

![png](modulation_classification_cnn/output_54_34.png)

Accuracy at SNR = 2 is 0.7900000214576721 %

![png](modulation_classification_cnn/output_54_36.png)

Accuracy at SNR = 4 is 0.800000011920929 %

![png](modulation_classification_cnn/output_54_38.png)

Accuracy at SNR = 6 is 0.8100000023841858 %

![png](modulation_classification_cnn/output_54_40.png)

Accuracy at SNR = 8 is 0.8100000023841858 %

![png](modulation_classification_cnn/output_54_42.png)

#### Combined Features

| Parameter           | Value  |
| ------------------- | ------ |
| Training loss       | 0.9299 |
| Training accuracy   | 0.6202 |
| Validation loss     | 1.0782 |
| Validation accuracy | 0.5641 |
| Number of epochs    | 16     |

![png](modulation_classification_cnn/output_65_2.png)

![png](modulation_classification_cnn/output_65_3.png)

```text

            8PSK       0.70      0.52      0.60     36000
          AM-DSB       0.47      0.75      0.58     36000
            BPSK       0.54      0.67      0.60     36000
           CPFSK       0.74      0.64      0.68     36000
            GFSK       0.57      0.71      0.63     36000
            PAM4       0.82      0.71      0.76     36000
           QAM16       0.59      0.48      0.53     36000
           QAM64       0.64      0.51      0.56     36000
            QPSK       0.47      0.59      0.52     36000
            WBFM       0.51      0.29      0.37     36000

        accuracy                           0.59    360000
       macro avg       0.60      0.59      0.58    360000
    weighted avg       0.60      0.59      0.58    360000


```

- Most confused class: QPSK
- Accuracy at SNR = 0 is 0.81

![png](modulation_classification_cnn/output_67_1.png)

![png](modulation_classification_cnn/output_67_2.png)

![png](modulation_classification_cnn/output_67_3.png)

![png](modulation_classification_cnn/output_67_4.png)

![png](modulation_classification_cnn/output_67_5.png)

![png](modulation_classification_cnn/output_67_6.png)

![png](modulation_classification_cnn/output_67_7.png)

![png](modulation_classification_cnn/output_67_8.png)

![png](modulation_classification_cnn/output_67_9.png)

![png](modulation_classification_cnn/output_67_10.png)

![png](modulation_classification_cnn/output_67_11.png)

![png](modulation_classification_cnn/output_67_12.png)

![png](modulation_classification_cnn/output_67_13.png)

![png](modulation_classification_cnn/output_67_14.png)

![png](modulation_classification_cnn/output_67_15.png)

![png](modulation_classification_cnn/output_67_16.png)

![png](modulation_classification_cnn/output_67_17.png)

![png](modulation_classification_cnn/output_67_18.png)

![png](modulation_classification_cnn/output_67_19.png)

![png](modulation_classification_cnn/output_67_20.png)

![png](modulation_classification_cnn/output_67_21.png)

![png](modulation_classification_cnn/output_67_22.png)

#### Embedded Features (SNR)

| Parameter           | Value  |
| ------------------- | ------ |
| Training loss       | 0.8305 |
| Training accuracy   | 0.6567 |
| Validation loss     | 0.9309 |
| Validation accuracy | 0.6185 |
| Number of epochs    | 15     |

![png](modulation_classification_cnn/output_76_2.png)

![png](modulation_classification_cnn/output_76_3.png)

```text
                  precision    recall  f1-score   support

            8PSK       0.62      0.52      0.57     36000
          AM-DSB       0.48      0.77      0.59     36000
            BPSK       0.58      0.71      0.64     36000
           CPFSK       0.79      0.65      0.71     36000
            GFSK       0.81      0.65      0.72     36000
            PAM4       0.87      0.74      0.80     36000
           QAM16       0.60      0.57      0.58     36000
           QAM64       0.51      0.82      0.63     36000
            QPSK       0.66      0.56      0.61     36000
            WBFM       0.60      0.29      0.39     36000

        accuracy                           0.63    360000
       macro avg       0.65      0.63      0.62    360000
    weighted avg       0.65      0.63      0.62    360000


```

- Most confused class: AM-DSB, WBFM
- Accuracy at SNR = 0 is 0.89

![png](modulation_classification_cnn/output_77_1.png)

![png](modulation_classification_cnn/output_77_2.png)

![png](modulation_classification_cnn/output_77_3.png)

![png](modulation_classification_cnn/output_77_4.png)

![png](modulation_classification_cnn/output_77_5.png)

![png](modulation_classification_cnn/output_77_6.png)

![png](modulation_classification_cnn/output_77_7.png)

![png](modulation_classification_cnn/output_77_8.png)

![png](modulation_classification_cnn/output_77_9.png)

![png](modulation_classification_cnn/output_77_10.png)

![png](modulation_classification_cnn/output_77_11.png)

![png](modulation_classification_cnn/output_77_12.png)

![png](modulation_classification_cnn/output_77_13.png)

![png](modulation_classification_cnn/output_77_14.png)

![png](modulation_classification_cnn/output_77_15.png)

![png](modulation_classification_cnn/output_77_16.png)

![png](modulation_classification_cnn/output_77_17.png)

![png](modulation_classification_cnn/output_77_18.png)

![png](modulation_classification_cnn/output_77_19.png)

![png](modulation_classification_cnn/output_77_20.png)

![png](modulation_classification_cnn/output_77_21.png)

![png](modulation_classification_cnn/output_77_22.png)

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

#### With Integral

- Learning rate scheduler: step decay
- Learning rate start: 0.0009

| Parameter           | Value  |
| ------------------- | ------ |
| Training loss       | 1.1069 |
| Training accuracy   | 0.5341 |
| Validation loss     | 1.1392 |
| Validation accuracy | 0.5242 |
| Test accuracy       | 0.5242 |
| Number of epochs    | 80     |

![RNN Integral](img/rnn-int-acc.png)
![RNN Integral](img/rnn-int-SNRacc.png)

#### Combination Raw and Derivative

- Learning rate scheduler: step decay
- Learning rate start: 0.0009

| Parameter           | Value  |
| ------------------- | ------ |
| Training loss       | 1.0770 |
| Training accuracy   | 0.5427 |
| Validation loss     | 1.1412 |
| Validation accuracy | 0.5255 |
| Test accuracy       | 0.5237 |
| Number of epochs    | 57     |

![RNN Raw Derivative](img/rnn-rawdiff-acc.png)
![RNN Raw Derivative](img/rnn-rawdiff-SNRacc.png)

#### Combination Raw and Integral

- Learning rate scheduler: step decay
- Learning rate start: 0.0009

| Parameter           | Value  |
| ------------------- | ------ |
| Training loss       | 1.1493 |
| Training accuracy   | 0.5151 |
| Validation loss     | 1.1694 |
| Validation accuracy | 0.5106 |
| Test accuracy       | 0.5103 |
| Number of epochs    | 34     |

![RNN Raw Integral](img/rnn-rawint-acc.png)
![RNN Raw Integral](img/rnn-rawint-SNRacc.png)

#### Combination Derivative and Integral

- Learning rate scheduler: step decay
- Learning rate start: 0.0009

| Parameter           | Value  |
| ------------------- | ------ |
| Training loss       | 1.1084 |
| Training accuracy   | 0.5318 |
| Validation loss     | 1.1370 |
| Validation accuracy | 0.5273 |
| Test accuracy       | 0.5281 |
| Number of epochs    | 65     |

![RNN Derivative Integral](img/rnn-diffint-acc.png)
![RNN Derivative Integral](img/rnn-diffint-SNRacc.png)



### LSTM model


#### Experiment 1

```text
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lstm_1 (LSTM)                (None, 22)                13288     
_________________________________________________________________
dense_3 (Dense)              (None, 10)                230       
=================================================================
Total params: 13,518
Trainable params: 13,518
Non-trainable params: 0
_________________________________________________________________
```
Learning rate scheduler: step decay
Learning rate: 0.0009

| Parameter           | Value  |
| ------------------- | ------ |
| Training loss       | 1.487  |
| Training accuracy   | 0.389  |
| Validation loss     | 1.4979 |
| Validation accuracy | 0.3837 |
| Number of epochs    | 100    |

![Experiment1 ](img/lstmexp1.PNG)

#### Experiment 2

```text
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lstm (LSTM)                  (None, 128)               131584    
_________________________________________________________________
dense (Dense)                (None, 64)                8256      
_________________________________________________________________
dense_1 (Dense)              (None, 32)                2080      
_________________________________________________________________
dropout (Dropout)            (None, 32)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 10)                330       
=================================================================
Total params: 142,250
Trainable params: 142,250
Non-trainable params: 0
_________________________________________________________________

```
Learning rate scheduler: step decay
Learning rate: 0.0001

| Parameter           | Value  |
| ------------------- | ------ |
| Training loss       | 1.1282 |
| Training accuracy   | 0.5082 |
| Validation loss     | 1.1718 |
| Validation accuracy | 0.4999 |
| Number of epochs    | 100    |

![Experiment2](img/lstmexp2.PNG)

#### Experiment 3 (final version)
```text
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lstm (LSTM)                  (None, 256)               394240    
_________________________________________________________________
dense (Dense)                (None, 128)               32896     
_________________________________________________________________
dense_1 (Dense)              (None, 64)                8256      
_________________________________________________________________
dense_2 (Dense)              (None, 32)                2080      
_________________________________________________________________
dropout (Dropout)            (None, 32)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 10)                330       
=================================================================
Total params: 437,802
Trainable params: 437,802
Non-trainable params: 0
_________________________________________________________________
```

- Learning rate scheduler: step decay
- Learning rate start: 0.0009

| Parameter           | Value  |
| ------------------- | ------ |
| Training loss       | 1.0844 |
| Training accuracy   | 0.5291 |
| Validation loss     | 1.3189 |
| Validation accuracy | 0.5200 |
| Number of epochs    | 100    |

![Experiment3](img/lstmexp3.PNG)

```text
 precision    recall  f1-score   support

        8PSK       0.32      0.45      0.37     36000
      AM-DSB       0.47      0.79      0.59     36000
        BPSK       0.42      0.73      0.53     36000
       CPFSK       0.69      0.62      0.65     36000
        GFSK       0.75      0.68      0.72     36000
        PAM4       0.86      0.64      0.73     36000
       QAM16       0.42      0.29      0.34     36000
       QAM64       0.57      0.40      0.47     36000
        QPSK       0.34      0.31      0.32     36000
        WBFM       0.83      0.26      0.40     36000

    accuracy                           0.52    360000
   macro avg       0.57      0.52      0.51    360000
weighted avg       0.57      0.52      0.51    360000
```

![PNG](img/lstmexp3snr1.PNG)
![PNG](img/lstmexp3snr2.PNG)
![PNG](img/lstmexp3snr3.PNG)
![PNG](img/lstmexp3snr4.PNG)
![PNG](img/lstmexp3snr5.PNG)
![PNG](img/lstmexp3snr6.PNG)
![PNG](img/lstmexp3snr7.PNG)
![PNG](img/lstmexp3snr8.PNG)
![PNG](img/lstmexp3snr9.PNG)
![PNG](img/lstmexp3snr10.PNG)
![PNG](img/lstmexp3snr11.PNG)


#### With First Derivative

- Learning rate scheduler: step decay
- Learning rate start: 0.0009
-first experiment architechture was used

| Parameter           | Value  |
| ------------------- | ------ |
| Training loss       | 1.7470 |
| Training accuracy   | 0.3135 |
| Validation loss     | 1.7604 |
| Validation accuracy | 0.3085 |
| Test accuracy       | 0.31   |
| Number of epochs    | 100    |

![Derivative](img/lstmdiff.PNG) 

```text
          precision    recall  f1-score   support

        8PSK       0.15      0.07      0.09     36000
      AM-DSB       0.36      0.50      0.42     36000
        BPSK       0.15      0.14      0.14     36000
       CPFSK       0.32      0.59      0.41     36000
        GFSK       0.38      0.63      0.47     36000
        PAM4       0.47      0.46      0.46     36000
       QAM16       0.22      0.04      0.07     36000
       QAM64       0.26      0.51      0.34     36000
        QPSK       0.17      0.03      0.05     36000
        WBFM       0.29      0.13      0.18     36000

    accuracy                           0.31    360000
   macro avg       0.28      0.31      0.27    360000
weighted avg       0.28      0.31      0.27    360000
```


![PNG](img/lstmdiff1.PNG)
![PNG](img/lstmdiff2.PNG)
![PNG](img/lstmdiff3.PNG)
![PNG](img/lstmdiff4.PNG)
![PNG](img/lstmdiff5.PNG)
![PNG](img/lstmdiff6.PNG)
![PNG](img/lstmdiff7.PNG)
![PNG](img/lstmdiff8.PNG)
![PNG](img/lstmdiff9.PNG)
![PNG](img/lstmdiff10.PNG)
![PNG](img/lstmdiff11.PNG)

#### With Integral

- Learning rate scheduler: step decay
- Learning rate start: 0.0009
- second experiment was used


| Parameter           | Value  |
| ------------------- | ------ |
| Training loss       | 1.0711 |
| Training accuracy   | 0.5427 |
| Validation loss     | 1.1411 |
| Validation accuracy | 0.5288 |
| Test accuracy       | 0.53   |
| Number of epochs    | 100    |

![Integral](img/lstmint.PNG)

```text
precision    recall  f1-score   support

        8PSK       0.26      0.52      0.35     36000
      AM-DSB       0.47      0.80      0.59     36000
        BPSK       0.57      0.65      0.60     36000
       CPFSK       0.66      0.62      0.64     36000
        GFSK       0.69      0.70      0.70     36000
        PAM4       0.85      0.67      0.75     36000
       QAM16       0.34      0.13      0.19     36000
       QAM64       0.53      0.57      0.55     36000
        QPSK       0.51      0.37      0.43     36000
        WBFM       0.85      0.25      0.39     36000

    accuracy                           0.53    360000
   macro avg       0.57      0.53      0.52    360000
weighted avg       0.57      0.53      0.52    360000
```

![PNG](img/lstmdintsnr1.PNG)
![PNG](img/lstmintsnr2.PNG)
![PNG](img/lstmintsnr3.PNG)
![PNG](img/lstmintsnr4.PNG)
![PNG](img/lstmintsnr5.PNG)
![PNG](img/lstmintsnr6.PNG)
![PNG](img/lstmintsnr7.PNG)
![PNG](img/lstmintsnr8.PNG)
![PNG](img/lstmintsnr9.PNG)
![PNG](img/lstmintsnr10.PNG)
![PNG](img/lstmintsnr11.PNG)

#### Combination Raw and Derivative

- Learning rate scheduler: step decay
- Learning rate start: 0.0009

| Parameter           | Value  |
| ------------------- | ------ |
| Training loss       | 0.9455 |
| Training accuracy   | 0.5914 |
| Validation loss     | 1.5697 |
| Validation accuracy | 0.5253 |
| Test accuracy       | 0.58   |
| Number of epochs    | 100    |

![Raw Derivative](img/lstmrd.PNG)

```text
           precision    recall  f1-score   support

        8PSK       0.29      0.45      0.35     36000
      AM-DSB       0.46      0.80      0.58     36000
        BPSK       0.51      0.64      0.57     36000
       CPFSK       0.63      0.62      0.63     36000
        GFSK       0.75      0.67      0.71     36000
        PAM4       0.83      0.67      0.74     36000
       QAM16       0.39      0.26      0.31     36000
       QAM64       0.54      0.44      0.48     36000
        QPSK       0.43      0.42      0.43     36000
        WBFM       0.83      0.26      0.40     36000

    accuracy                           0.58    360000
   macro avg       0.57      0.52      0.58    360000
weighted avg       0.57      0.52      0.57    360000
```

![PNG](img/lstmrdsnr1.PNG)
![PNG](img/lstmrdsnr2.PNG)
![PNG](img/lstmrdsnr3.PNG)
![PNG](img/lstmrdsnr4.PNG)
![PNG](img/lstmrdsnr5.PNG)
![PNG](img/lstmrdsnr6.PNG)
![PNG](img/lstmrdsnr7.PNG)
![PNG](img/lstmrdsnr8.PNG)
![PNG](img/lstmrdsnr9.PNG)
![PNG](img/lstmrdsnr10.PNG)
![PNG](img/lstmrdsnr11.PNG)

#### Combination Raw and Integral

- Learning rate scheduler: step decay
- Learning rate start: 0.0009

| Parameter           | Value  |
| ------------------- | ------ |
| Training loss       | 0.8854 |
| Training accuracy   | 0.6175 |
| Validation loss     | 2.4617 |
| Validation accuracy | 0.5198 |
| Test accuracy       | 0.52   |
| Number of epochs    | 100    |

![Raw Integral](img/rnn-rawint-acc.PNG)

```text
          precision    recall  f1-score   support

        8PSK       0.26      0.51      0.35     36000
      AM-DSB       0.46      0.80      0.58     36000
        BPSK       0.59      0.62      0.60     36000
       CPFSK       0.70      0.58      0.63     36000
        GFSK       0.67      0.69      0.68     36000
        PAM4       0.81      0.66      0.73     36000
       QAM16       0.39      0.26      0.31     36000
       QAM64       0.48      0.41      0.44     36000
        QPSK       0.50      0.37      0.42     36000
        WBFM       0.80      0.26      0.39     36000

    accuracy                           0.52    360000
   macro avg       0.57      0.52      0.51    360000
weighted avg       0.57      0.52      0.51    360000
```

![PNG](img/lstmrisnr1.PNG)
![PNG](img/lstmrisnr2.PNG)
![PNG](img/lstmrisnr3.PNG)
![PNG](img/lstmrisnr4.PNG)
![PNG](img/lstmrisnr5.PNG)
![PNG](img/lstmrisnr6.PNG)
![PNG](img/lstmrisnr7.PNG)
![PNG](img/lstmrisnr8.PNG)
![PNG](img/lstmrisnr9.PNG)
![PNG](img/lstmrisnr10.PNG)
![PNG](img/lstmrisnr11.PNG)

#### Combination Derivative and Integral

- Learning rate scheduler: step decay
- Learning rate start: 0.0009

| Parameter           | Value  |
| ------------------- | ------ |
| Training loss       | 0.9326 |
| Training accuracy   | 0.6006 |
| Validation loss     | 1.8830 |
| Validation accuracy | 0.5185 |
| Test accuracy       | 0.52   |
| Number of epochs    | 100    |

![Derivative Integral](img/lstmdi.PNG)

```text
          precision    recall  f1-score   support

        8PSK       0.26      0.51      0.35     36000
      AM-DSB       0.46      0.80      0.58     36000
        BPSK       0.59d      0.62      0.60     36000
       CPFSK       0.70      0.58      0.63     36000
        GFSK       0.67      0.69      0.68     36000
        PAM4       0.81      0.66      0.73     36000
       QAM16       0.39      0.26      0.31     36000
       QAM64       0.48      0.41      0.44     36000
        QPSK       0.50      0.37      0.42     36000
        WBFM       0.80      0.26      0.39     36000

    accuracy                           0.52    360000
   macro avg       0.57      0.52      0.51    360000
weighted avg       0.57      0.52      0.51    360000
```

![PNG](img/lstmdisnr1.PNG)
![PNG](img/lstmdisnr2.PNG)
![PNG](img/lstmdisnr3.PNG)
![PNG](img/lstmdisnr4.PNG)
![PNG](img/lstmdisnr5.PNG)
![PNG](img/lstmdisnr6.PNG)
![PNG](img/lstmdisnr7.PNG)
![PNG](img/lstmdisnr8.PNG)
![PNG](img/lstmdisnr9.PNG)
![PNG](img/lstmdisnr10.PNG)
![PNG](img/lstmdisnr11.PNG)

<!-- References -->
[github]: github.com/moharamfatema/modulation-classification
[github-badge]: https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white

[license]: https://github.com/moharamfatema/modulation-classification/blob/main/COPYING

[license-badge]: https://img.shields.io/badge/License-GPLv3-blue?style=for-the-badge

[binder-badge]: https://img.shields.io/badge/launch-binder-orange.svg?style=for-the-badge

[binder]: https://mybinder.org/v2/gh/moharamfatema/modulation-classification/main

[cnn-model]: https://github.com/moharamfatema/modulation-classification/blob/main/docs/img/cnn-model.png
