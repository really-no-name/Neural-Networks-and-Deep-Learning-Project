# Neural Networks and Deep Learning Project

**Important: Prohibit the use of this project for coursework purposes!**

**Author: Bolun Xu**

---

## Project Overview

This project is focused on utilizing The CIFAR-10 dataset for image classification and is designed to run on Google Colab.

## Neural Network Structure

The core of this project is a Convolutional Neural Network (CNN), designed specifically to classify images within the CIFAR-10 dataset. This dataset comprises 10 categories of 32x32 color images. The network structure is sequentially organized and includes the following components:

1. **Convolutional Layers (卷积层)**
    - The first layer is a convolutional layer, featuring 64 filters (3x3 kernel size) with a 1-pixel padding to maintain the image size.
    - The second layer has 128 filters, maintaining the same kernel size and padding.
    - The third layer includes 256 filters, consistent in kernel size and padding with the previous layers.
    - These layers apply filters to capture image features. Each is followed by Batch Normalization and ReLU activation functions, facilitating the learning of non-linear features and speeding up training.

2. **Max Pooling Layers (最大池化层)**
    - Following each convolutional layer is a Max Pooling Layer with a 2x2 pooling region and a stride of 2, designed to reduce feature map size while preserving critical features.

3. **Flatten Layer (展平层)**
    - Post convolutional layers, the feature map is flattened into a one-dimensional vector for processing by fully connected layers.

4. **Fully Connected Layers (全连接层)**
    - A fully connected layer with 1024 neurons follows, including a ReLU activation function and a Dropout operation (with a dropout ratio of 0.6) to mitigate overfitting and enhance network generalization.
    - The final layer is a fully connected layer with 10 output units, corresponding to the CIFAR-10 dataset's 10 categories. It generates logits for classification.

The network employs the CrossEntropyLoss function for training—a standard for multi-class classification. Adam optimizer, known for its adaptive learning rate, is used for optimizing the model. Additionally, a learning rate decay strategy is implemented to gradually reduce the learning rate during training, aiding in better model convergence.
