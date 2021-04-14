from model import *
from dataset import *
import numpy as np
from time import time
import random

model_info = {"conv1" : 16,
                    "conv2_1_1" : 16,
                    "conv2_1_2" : 16,
                    "conv2_2_1" : 16,
                    "conv2_2_2" : 16,
                    "conv2_3_1" : 16,
                    "conv2_3_2" : 16,
                    "conv3_1_1" : 32,
                    "conv3_1_2" : 32,
                    "conv3_2_1" : 32,
                    "conv3_2_2" : 32,
                    "conv3_3_1" : 32,
                    "conv3_3_2" : 32,
                    "conv4_1_1" : 64,
                    "conv4_1_2" : 64,
                    "conv4_2_1" : 64,
                    "conv4_2_2" : 64,
                    "conv4_3_1" : 64,
                    "conv4_3_2" : 64,
                    "fc1" : 10,}

   
if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels = load_data()
    
    # Construct
    resnet = ResNet20()
    resnet.build(model_info)
#    resnet.load("\model\prune_50.ckpt")
    

#訓練最好的原始模型
    resnet.sess.run(tf.assign(resnet.learning_rate,1e-3))
    max_accuracy = resnet.test(test_images, test_labels)
    print("模組準確率：",max_accuracy)               
    total_time = 0
    epochs = 300
    for index in range(epochs):
        if index==epochs/3:
            resnet.sess.run(tf.assign(resnet.learning_rate,1e-4))
        if index==epochs*2/3:
            resnet.sess.run(tf.assign(resnet.learning_rate,1e-5))            
        
        print("=========================開始第{}次訓練=========================".format(index+1))
        start = time()
        print("開始做影像增強")
#==========================================================================================================       
        datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0.,
        # set range for random channel shifts
        channel_shift_range=0.,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)
    
        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
    #    datagen.fit(train_images)
        data = datagen.flow(train_images, train_labels,batch_size=20)


        data_images = []
        data_labels = []
        for i in range(0,2500):
            data_images.append(data[i][0])     
            data_labels.append(data[i][1]) 
            
        print("開始訓練")        
#        print(resnet.sess.run(resnet.learning_rate))
        pred = resnet.train(data_images, data_labels,epoch = 1)
        accuracy = resnet.test(test_images, test_labels)        
        if accuracy > max_accuracy:              
            max_accuracy = accuracy
            resnet.save(r'\best_mdel.ckpt')

        print("第{}次訓練，模組準確率{}，目前最好準確率{}".format(index,accuracy,max_accuracy))
        total_time+=time()-start
        print('單次訓練+測試時間(到目前為止平均)：',total_time/(index+1),"\n\n")
    

    
    
        







