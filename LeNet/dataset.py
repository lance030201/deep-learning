import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.utils import shuffle
from keras.datasets import mnist
from keras.utils import np_utils
import cv2

def load_data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# =============================================================================
#     mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#     train_images = mnist.train.images
#     train_labels = mnist.train.labels
#     test_images = mnist.test.images
#     test_labels = mnist.test.labels
# 
#     
#     train_images = np.reshape(train_images,(55000,28,28,1))
#     test_images = np.reshape(test_images,(10000,28,28,1))
# =============================================================================
    
    train_images = np.reshape(train_images,(60000,28,28,1))
    test_images = np.reshape(test_images,(10000,28,28,1))
    train_labels = np.reshape(train_labels,(60000,1))
    test_labels = np.reshape(test_labels,(10000,1))
    train_labels = np_utils.to_categorical(train_labels)
    test_labels = np_utils.to_categorical(test_labels)



#    cv2.imshow("train",train_images[0])
#    cv2.imshow("test",test_images[0])
#    print(test_labels[0])
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    
    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')
    
    train_images[:, :, :, 0] = (train_images[:, :, :, 0] - np.mean(train_images[:, :, :, 0])) / np.std(train_images[:, :, :, 0])
    test_images[:, :, :, 0] = (test_images[:, :, :, 0] - np.mean(test_images[:, :, :, 0])) / np.std(test_images[:, :, :, 0])
    train_images, train_labels = shuffle(train_images, train_labels, random_state=0)
    test_images, test_labels = shuffle(test_images, test_labels, random_state=0)
    
#    print(train_images.shape)
#    print(train_labels.shape)
#    print(test_images.shape)
#    print(test_labels.shape)
    return train_images, train_labels, test_images, test_labels
    
#load_data()
    
    





