# Deep Learning
Some deep learning model implementation and result

- [Environment](#environment)
- [What is my code doing](#what-is-my-code-doing)
- [Experiment](#experiment)
  * [LeNet](#lenet)
  * [AlexNet](#alexnet)
  * [VGG16](#vgg16)
  * [ResNet](#resnet)

## Environment
**Python**       3.5    
**tensorflow**   1.10    
**CPU**          Intel i7-7700    
**GPU**          NVIDIA GeForce GTX 1080Ti    

## What is my code doing
**train.py**       training model    
**model.py**       deep model    
**load_test.py**   test the result    
**dataset.py**     dataset preprocessing    

## Experiment
### LeNet

| dataset | epochs | accuracy | batch normalization | dropout |
| :-----: | :----: | :------: | :-----------------: | :-----: | 
| MNIST   | 90     | 99.44    | O                   | 0.5     |

### AlexNet

| dataset | epochs | accuracy | batch normalization | dropout |
| :-----: | :----: | :------: | :-----------------: | :-----: | 
| CIFAR10 | 150    | 79.2     | O                   | 0.5     |

### VGG16

| dataset | epochs | accuracy | batch normalization | dropout |
| :-----: | :----: | :------: | :-----------------: | :-----: | 
| CIFAR10 | 300    | 91.66    | O                   | 0.5     |

### ResNet20

| dataset | epochs | accuracy | batch normalization | dropout |
| :-----: | :----: | :------: | :-----------------: | :-----: | 
| CIFAR10 | 300    | 90.42    | O                   | 0.5     |












