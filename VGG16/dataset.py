import numpy as np
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.utils import to_categorical
from sklearn.utils import shuffle

def load_data():
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    
    train_labels = np_utils.to_categorical(train_labels, num_classes = 10)
    test_labels = np_utils.to_categorical(test_labels, num_classes = 10)
    
    #train_images = np.delete(train_images, np.s_[49000:50000], 0)  # 删除A的第二行
    #train_labels = np.delete(train_labels, np.s_[49000:50000], 0)  # 删除A的第二行

    #train_images = np.append(train_images,test_images[0:1000],0)
    #train_labels = np.append(train_labels,test_labels[0:1000],0)
    
    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')
    
    train_images[:, :, :, 0] = (train_images[:, :, :, 0] - np.mean(train_images[:, :, :, 0])) / np.std(train_images[:, :, :, 0])
    train_images[:, :, :, 1] = (train_images[:, :, :, 1] - np.mean(train_images[:, :, :, 1])) / np.std(train_images[:, :, :, 1])
    train_images[:, :, :, 2] = (train_images[:, :, :, 2] - np.mean(train_images[:, :, :, 2])) / np.std(train_images[:, :, :, 2])

    test_images[:, :, :, 0] = (test_images[:, :, :, 0] - np.mean(test_images[:, :, :, 0])) / np.std(test_images[:, :, :, 0])
    test_images[:, :, :, 1] = (test_images[:, :, :, 1] - np.mean(test_images[:, :, :, 1])) / np.std(test_images[:, :, :, 1])
    test_images[:, :, :, 2] = (test_images[:, :, :, 2] - np.mean(test_images[:, :, :, 2])) / np.std(test_images[:, :, :, 2])
    
    
    train_images, train_labels = shuffle(train_images, train_labels, random_state=0)
    test_images, test_labels = shuffle(test_images, test_labels, random_state=0)
    
    return train_images, train_labels, test_images, test_labels

    