import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import numpy as np
import math
import cv2
import os
import time

def horizontal_flip(image, axis):
    #axis 0 垂直翻转，1水平翻转 ，-1水平垂直翻转，2不翻转，各自以25%的可能性
    if axis != 0:
        image = cv2.flip(image, 1)
    return image
    
def random_flip(batch_data):
    #帮助随机翻转一批图像
    flip_batch = np.zeros(len(batch_data) * 32* 32 * 3).reshape(len(batch_data), 32, 32, 3)
    for i in range(len(batch_data)):
        axis1 = np.random.randint(low=0, high=2)
        flip_batch[i, ...] = horizontal_flip(image=batch_data[i, ...], axis=axis1)
    return flip_batch

class ResNet20():
    def __init__(self):
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"    
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)  
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#        self.sess = tf.Session()
        
        self.model_info = {"conv1" : 16,
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
        
        
        self.model_variables = {}        
        self.norm = True
        
# =============================================================================
#         self.learning_rate = tf.Variable(float(0.001), trainable=False, dtype=tf.float32)
#         self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate*0.1)  
# =============================================================================
        
        
        
    def reset(self):
        tf.reset_default_graph()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)  
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#        self.sess = tf.Session()
    
    def save(self, path = r'\model\save_output.ckpt'):
        self.printTrainableVariables()

        isSave = False
        saver=tf.train.Saver()
        try:
            print(os.getcwd() + path)
            saver.save(self.sess,os.getcwd() + path) #文件名根據自己的實際情況進行修改
            self.writeInfoToTxt()
            print('模型保存成功')
            isSave = True
        except :
            print('保存文件失敗，請設置好路徑，確只路徑存在')
        return isSave

        
    def load(self, path = r'\model\save.ckpt'):
        isRestore = False
        saver=tf.train.Saver()
        try:
            saver.restore(self.sess,os.getcwd() + path) #文件名根據自己的實際情況進行修改
            self.model_variables = self.getVariables()
            print('調用文件成功！')
            isRestore = True
        except:
            print('調用文件失敗，請確保模型文件存在！')
        return isRestore
    
    def loadNpy(self,path):
        for var in tf.trainable_variables():
            if 'conv' in var.name or 'fc' in var.name:
                read = path+var.name+'.npy'
                read = read.replace('/','_')
                read = read.replace(':','_')
                update=tf.assign(var,np.load(read))
            
            self.sess.run(update)

    

    
    def build(self, model_info = None):
        if(model_info == None):
            model_info = self.model_info.copy()
        else :
            self.model_info = model_info.copy()
            
#        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
#        self.sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=self.gpu_options))
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)  
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#        self.sess = tf.Session()
        
        
# =============================================================================
#         self.augmentation_in = tf.placeholder(tf.float32, [None, 32, 32, 3])
#         self.augmentation_in = tf.image.random_flip_up_down(self.augmentation_in)
#         self.augmentation_in = tf.image.random_flip_left_right(self.augmentation_in)
#         
#         self.augmentation_in = tf.image.resize_image_with_crop_or_pad(self.augmentation_in,36,36)
#         
#         #self.augmentation_in= tf.reshape(self.augmentation_in, [36, 36, 3, None])
#         self.augmentation_in = tf.random_crop(self.augmentation_in,[100,32,32,3])
#         
#         self.augmentation_in = tf.image.random_brightness(self.augmentation_in, max_delta=0.05)
#         self.augmentation_out = tf.image.random_contrast(self.augmentation_in, 0.8, 1.2)
#         self.augmentation_out = tf.image.random_hue(self.augmentation_in, 0.08)
#         #self.augmentation_out = tf.image.random_saturation(self.augmentation_in, 0.6, 1.4)
# =============================================================================
        
        
        
        
        
    
        #紀錄現在的值,(?, 2)
        self.parameters = []
        
        
        self.imgs = tf.placeholder(tf.float32, [None, 32, 32, 3])
        self.labs = tf.placeholder(tf.float32, [None, self.model_info["fc1"]])
        self.keep_prob_conv_1 = tf.placeholder(tf.float32)
        self.keep_prob_conv = tf.placeholder(tf.float32)
        self.keep_prob = tf.placeholder(tf.float32)
        self.conv1 = 0
        
# =============================================================================
#         with tf.name_scope("setting") as scope:
#             self.learning_rate = tf.Variable(float(0.001), trainable=False, dtype=tf.float32)
#             self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate*0.1) 
# =============================================================================
        
        self.learning_rate = tf.Variable(float(0.001), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate*0.1) 
        
        # conv1
        with tf.name_scope('conv1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, model_info['conv1']], dtype=tf.float32,
                                                     stddev=1e-2), name='weights')
#            kernel = tf.get_variable(name="weights",initializer=tf.truncated_normal([3, 3, 3, model_info['conv1']], dtype=tf.float32,stddev=1e-2),regularizer=self.regularizer)
            conv = tf.nn.conv2d(self.imgs, kernel, [1, 1, 1, 1], padding='SAME')
            
            biases = tf.Variable(tf.constant(0.0, shape=[model_info['conv1']], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            
#            regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
#            out = tf.layers.conv2d(self.imgs, model_info['conv1'], 3, kernel_initializer="truncated_normal", kernel_regularizer=regularizer,strides=(1, 1), name='weights',padding='same')
            
            # Batch Normalize
            '''
            img = tf.Variable(tf.random_normal([3, 32, 32, 64]))
            print("--------------------")
            axis = list(range(len(img.get_shape())-1))
            
            print(img.get_shape())
            print(len(img.get_shape())-1)
            print(list(range(len(img.get_shape())-1)))'''
            
            fc_mean, fc_var = tf.nn.moments(
                out,
                axes=[0,1,2],   # the dimension you wanna normalize, here [0] for batch
                            # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
            )
            scale = tf.Variable(tf.ones([model_info['conv1']]), name='norms/scale')
            shift = tf.Variable(tf.zeros([model_info['conv1']]), name='norms/shift')
            epsilon = 0.001
            # apply moving average for mean and var when train on batch
            ema = tf.train.ExponentialMovingAverage(decay=0.5)
            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean), tf.identity(fc_var)
            mean, var = mean_var_with_update()
            
#            out = BatchNormalization()(out)
            out = tf.nn.batch_normalization(out, mean, var, shift, scale, epsilon)
            #-------------------------------------------------------------------
            
            self.conv1 = tf.nn.relu(out, name=scope)

#            print(self.conv1.get_shape())
#            self.resize_conv1 = self.feature_map_to_next_layer(self.conv1,"conv2_1_2")
#            self.conv1 = tf.nn.dropout(self.conv1, self.keep_prob)        
#            self.parameters.append([kernel, biases])
#            tf.add_to_collection(tf.GraphKeys.WEIGHTS, kernel)
#            tf.add_to_collection(tf.GraphKeys.WEIGHTS, biases)

        # conv2_1_1
        with tf.name_scope('conv2_1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, model_info['conv1'], model_info['conv2_1_1']], dtype=tf.float32,
                                                     stddev=1e-2), name='weights')
            conv = tf.nn.conv2d(self.conv1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[model_info['conv2_1_1']], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
#            regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
#            out = tf.layers.conv2d(self.conv1,model_info['conv2_1_1'],3,kernel_regularizer=regularizer,strides=(1, 1),padding='same')
            
            # Batch Normalize
            fc_mean, fc_var = tf.nn.moments(
                out,
                axes=[0,1,2],   # the dimension you wanna normalize, here [0] for batch
                            # for image, you wanna do [0, 1, 2] for [batch, heigyht, width] but not channel
            )
            scale = tf.Variable(tf.ones([model_info['conv2_1_1']]), name='norms/scale')
            shift = tf.Variable(tf.zeros([model_info['conv2_1_1']]), name='norms/shift')
            epsilon = 0.001
            # apply moving average for mean and var when train on batch
            ema = tf.train.ExponentialMovingAverage(decay=0.5)
            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean), tf.identity(fc_var)
            mean, var = mean_var_with_update()
            
            out = tf.nn.batch_normalization(out, mean, var, shift, scale, epsilon)
#            out = BatchNormalization()(out)
            #-------------------------------------------------------------------
            
            self.conv2_1_1 = tf.nn.relu(out, name=scope)
#            print(self.conv2_1_1.get_shape())
#            self.conv2_1_1 = tf.nn.dropout(self.conv2_1_1, self.keep_prob)  
#            self.parameters += [kernel, biases]
            self.parameters.append([kernel, biases])

# =============================================================================
#         # pool1
#         self.pool1 = tf.nn.max_pool(self.conv1_2,
#                                ksize=[1, 2, 2, 1],
#                                strides=[1, 2, 2, 1],
#                                padding='SAME',
#                                name='pool1')
# =============================================================================
        
        # conv2_1_2
        with tf.name_scope('conv2_1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, model_info['conv2_1_1'], model_info['conv2_1_2']], dtype=tf.float32,
                                                     stddev=1e-2), name='weights')
            conv = tf.nn.conv2d(self.conv2_1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[model_info['conv2_1_2']], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
#            regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
#            out = tf.layers.conv2d(self.conv2_1_1,model_info['conv2_1_2'],3,kernel_regularizer=regularizer,strides=(1, 1),padding='same')
            
            # Batch Normalize
            fc_mean, fc_var = tf.nn.moments(
                out,
                axes=[0,1,2],   # the dimension you wanna normalize, here [0] for batch
                            # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
            )
            scale = tf.Variable(tf.ones([model_info['conv2_1_2']]), name='norms/scale')
            shift = tf.Variable(tf.zeros([model_info['conv2_1_2']]), name='norms/shift')
            epsilon = 0.001
            # apply moving average for mean and var when train on batch
            ema = tf.train.ExponentialMovingAverage(decay=0.5)
            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean), tf.identity(fc_var)
            mean, var = mean_var_with_update()
            
#            out = BatchNormalization()(out)
            out = tf.nn.batch_normalization(out, mean, var, shift, scale, epsilon)
            #-------------------------------------------------------------------
            
            self.conv2_1_2 = out
            self.layer4_input = tf.add(self.conv1,self.conv2_1_2)
            self.layer4_input = tf.nn.relu(out, name=scope)
#            print(self.conv2_1_2.get_shape())
#            self.resize_conv2_1_2 = self.feature_map_to_next_layer(self.conv2_1_2,"conv2_2_2")
#            self.conv2_1_1 = tf.nn.dropout(self.conv2_1_2, self.keep_prob)  
#            self.parameters += [kernel, biases]
            self.parameters.append([kernel, biases])
            
        # conv2_2_1
        with tf.name_scope('conv2_2_1') as scope:
                        
            kernel = tf.Variable(tf.truncated_normal([3, 3, model_info['conv2_1_2'], model_info['conv2_2_1']], dtype=tf.float32,
                                                      stddev=1e-2), name='weights')
            conv = tf.nn.conv2d(self.layer4_input, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[model_info['conv2_2_1']], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
#            regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
#            out = tf.layers.conv2d(self.conv2_1_2,model_info['conv2_2_1'],3,kernel_regularizer=regularizer,strides=(1, 1),padding='same')
            
            # Batch Normalize
            fc_mean, fc_var = tf.nn.moments(
                out,
                axes=[0,1,2],   # the dimension you wanna normalize, here [0] for batch
                            # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
            )
            scale = tf.Variable(tf.ones([model_info['conv2_2_1']]), name='norms/scale')
            shift = tf.Variable(tf.zeros([model_info['conv2_2_1']]), name='norms/shift')
            epsilon = 0.001
            # apply moving average for mean and var when train on batch
            ema = tf.train.ExponentialMovingAverage(decay=0.5)
            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean), tf.identity(fc_var)
            mean, var = mean_var_with_update()
            
#            out = BatchNormalization()(out)
            out = tf.nn.batch_normalization(out, mean, var, shift, scale, epsilon)
            #-------------------------------------------------------------------
            
            self.conv2_2_1 = tf.nn.relu(out, name=scope)
#            print(self.conv2_2_1.get_shape())
#            self.conv2_2_1 = tf.nn.dropout(self.conv2_2_1, self.keep_prob)  
#            self.parameters += [kernel, biases]
            self.parameters.append([kernel, biases])
            
        # conv2_2_2
        with tf.name_scope('conv2_2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, model_info['conv2_2_1'], model_info['conv2_2_2']], dtype=tf.float32,
                                                     stddev=1e-2), name='weights')
            conv = tf.nn.conv2d(self.conv2_2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[model_info['conv2_2_2']], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
#            regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
#            out = tf.layers.conv2d(self.conv2_2_1,model_info['conv2_2_2'],3,kernel_regularizer=regularizer,strides=(1, 1), padding='same')
            
            # Batch Normalize
            fc_mean, fc_var = tf.nn.moments(
                out,
                axes=[0,1,2],   # the dimension you wanna normalize, here [0] for batch
                            # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
            )
            scale = tf.Variable(tf.ones([model_info['conv2_2_2']]), name='norms/scale')
            shift = tf.Variable(tf.zeros([model_info['conv2_2_2']]), name='norms/shift')
            epsilon = 0.001
            # apply moving average for mean and var when train on batch
            ema = tf.train.ExponentialMovingAverage(decay=0.5)
            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean), tf.identity(fc_var)
            mean, var = mean_var_with_update()
            
#            out = BatchNormalization()(out)
            out = tf.nn.batch_normalization(out, mean, var, shift, scale, epsilon)
            #-------------------------------------------------------------------
            
            self.conv2_2_2 = out
            self.layer6_input = tf.add(self.layer4_input,self.conv2_2_2)
            self.layer6_input = tf.nn.relu(self.layer6_input, name=scope)
#            print(self.conv2_2_2.get_shape())
#            self.resize_conv2_2_2 = self.feature_map_to_next_layer(self.conv2_2_2,"conv2_3_2")
#            self.conv2_2_2 = tf.nn.dropout(self.conv2_2_2, self.keep_prob)  
#            self.parameters += [kernel, biases]
            self.parameters.append([kernel, biases])
            
        # conv2_3_1
        with tf.name_scope('conv2_3_1') as scope:
            
            kernel = tf.Variable(tf.truncated_normal([3, 3, model_info['conv2_2_2'], model_info['conv2_3_1']], dtype=tf.float32,
                                                     stddev=1e-2), name='weights')
            conv = tf.nn.conv2d(self.layer6_input, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[model_info['conv2_3_1']], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
#            regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
#            out = tf.layers.conv2d(self.conv2_2_2,model_info['conv2_3_1'],3,kernel_regularizer=regularizer,strides=(1, 1), padding='same')
            
            # Batch Normalize
            fc_mean, fc_var = tf.nn.moments(
                out,
                axes=[0,1,2],   # the dimension you wanna normalize, here [0] for batch
                            # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
            )
            scale = tf.Variable(tf.ones([model_info['conv2_3_1']]), name='norms/scale')
            shift = tf.Variable(tf.zeros([model_info['conv2_3_1']]), name='norms/shift')
            epsilon = 0.001
            # apply moving average for mean and var when train on batch
            ema = tf.train.ExponentialMovingAverage(decay=0.5)
            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean), tf.identity(fc_var)
            mean, var = mean_var_with_update()
            
#            out = BatchNormalization()(out)
            out = tf.nn.batch_normalization(out, mean, var, shift, scale, epsilon)
#            -------------------------------------------------------------------
            
            self.conv2_3_1 = tf.nn.relu(out, name=scope)
#            print(self.conv2_3_1.get_shape())
#            self.conv2_3_1 = tf.nn.dropout(self.conv2_3_1, self.keep_prob)  
#            self.parameters += [kernel, biases]
            self.parameters.append([kernel, biases])
            
            # conv2_3_2
        with tf.name_scope('conv2_3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, model_info['conv2_3_1'], model_info['conv2_3_2']], dtype=tf.float32,
                                                     stddev=1e-2), name='weights')
            conv = tf.nn.conv2d(self.conv2_3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[model_info['conv2_3_2']], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
#            regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
#            out = tf.layers.conv2d(self.conv2_3_1,model_info['conv2_3_2'],3,kernel_regularizer=regularizer,strides=(1, 1), padding='same')
            
            # Batch Normalize
            fc_mean, fc_var = tf.nn.moments(
                out,
                axes=[0,1,2],   # the dimension you wanna normalize, here [0] for batch
                            # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
            )
            scale = tf.Variable(tf.ones([model_info['conv2_3_2']]), name='norms/scale')
            shift = tf.Variable(tf.zeros([model_info['conv2_3_2']]), name='norms/shift')
            epsilon = 0.001
            # apply moving average for mean and var when train on batch
            ema = tf.train.ExponentialMovingAverage(decay=0.5)
            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean), tf.identity(fc_var)
            mean, var = mean_var_with_update()
            
#            out = BatchNormalization()(out)
            out = tf.nn.batch_normalization(out, mean, var, shift, scale, epsilon)
            #-------------------------------------------------------------------
            self.conv2_3_2 = out
            self.layer8_input = tf.add(self.layer6_input,self.conv2_3_2)
            self.layer8_input = tf.nn.relu(self.layer8_input, name=scope)
            
            

#            self.resize_conv2_3_2 = tf.layers.conv2d(self.conv2_3_2,model_info['conv3_1_1'],1,kernel_regularizer=regularizer,strides=(2, 2), padding='same')
#            self.resize_conv2_3_2 = self.feature_map_to_next_layer(self.resize_conv2_3_2,"conv3_1_2")
#            self.conv2_3_2 = tf.nn.dropout(self.conv2_3_2, self.keep_prob)  
#            self.parameters += [kernel, biases]
            self.parameters.append([kernel, biases])
                        
#            out = tf.add(self.resize_conv2_2_2,out)            
            
#            print(self.conv2_3_2.get_shape())

        
        # conv3_1_1
        with tf.name_scope('conv3_1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, model_info['conv2_3_2'], model_info['conv3_1_1']], dtype=tf.float32,
                                                      stddev=1e-2), name='weights')
            conv = tf.nn.conv2d(self.layer8_input, kernel, [1, 2, 2, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[model_info['conv3_1_1']], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
#            regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
#            out = tf.layers.conv2d(self.conv2_3_2,model_info['conv3_1_1'],3,kernel_regularizer=regularizer,strides=(2, 2), padding='same')
            
            # Batch Normalize
            fc_mean, fc_var = tf.nn.moments(
                out,
                axes=[0,1,2],   # the dimension you wanna normalize, here [0] for batch
                            # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
            )
            scale = tf.Variable(tf.ones([model_info['conv3_1_1']]), name='norms/scale')
            shift = tf.Variable(tf.zeros([model_info['conv3_1_1']]), name='norms/shift')
            epsilon = 0.001
            # apply moving average for mean and var when train on batch
            ema = tf.train.ExponentialMovingAverage(decay=0.5)
            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean), tf.identity(fc_var)
            mean, var = mean_var_with_update()
            
#            out = BatchNormalization()(out)
            out = tf.nn.batch_normalization(out, mean, var, shift, scale, epsilon)
            #-------------------------------------------------------------------
            
            self.conv3_1_1 = tf.nn.relu(out, name=scope)
#            print(self.conv3_1_1.get_shape())
#            self.conv3_1_1 = tf.nn.dropout(self.conv3_1_1, self.keep_prob)  
#            self.parameters += [kernel, biases]
            self.parameters.append([kernel, biases])

# =============================================================================
#         # pool1
#         self.pool1 = tf.nn.max_pool(self.conv1_2,
#                                ksize=[1, 2, 2, 1],
#                                strides=[1, 2, 2, 1],
#                                padding='SAME',
#                                name='pool1')
# =============================================================================
        
        # conv3_1_2
        with tf.name_scope('conv3_1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, model_info['conv3_1_1'], model_info['conv3_1_2']], dtype=tf.float32,
                                                     stddev=1e-2), name='weights')
            conv = tf.nn.conv2d(self.conv3_1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[model_info['conv3_1_2']], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
#            regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
#            out = tf.layers.conv2d(self.conv3_1_1,model_info['conv3_1_2'],3,kernel_regularizer=regularizer,strides=(1, 1), padding='same')
            
            # Batch Normalize
            fc_mean, fc_var = tf.nn.moments(
                out,
                axes=[0,1,2],   # the dimension you wanna normalize, here [0] for batch
                            # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
            )
            scale = tf.Variable(tf.ones([model_info['conv3_1_2']]), name='norms/scale')
            shift = tf.Variable(tf.zeros([model_info['conv3_1_2']]), name='norms/shift')
            epsilon = 0.001
            # apply moving average for mean and var when train on batch
            ema = tf.train.ExponentialMovingAverage(decay=0.5)
            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean), tf.identity(fc_var)
            mean, var = mean_var_with_update()
            
#            out = BatchNormalization()(out)
            out = tf.nn.batch_normalization(out, mean, var, shift, scale, epsilon)
            #-------------------------------------------------------------------
            
#            out = tf.add(self.resize_conv2_3_2,out) 
            
            self.conv3_1_2 = out
            kernel_resize = tf.Variable(tf.truncated_normal([1, 1, model_info['conv2_3_2'], model_info['conv3_1_2']], dtype=tf.float32,
                                                     stddev=1e-2), name='weights')
            self.layer8_input_resize = tf.nn.conv2d(self.layer8_input, kernel_resize, [1, 2, 2, 1], padding='SAME')
            self.layer10_input = tf.add(self.layer8_input_resize,self.conv3_1_2)
            self.layer10_input = tf.nn.relu(self.layer10_input, name=scope)
#            print(self.conv3_1_2.get_shape())
#            self.resize_conv3_1_2 = self.feature_map_to_next_layer(self.conv3_1_2,"conv3_2_2")
#            self.conv3_1_2 = tf.nn.dropout(self.conv3_1_2, self.keep_prob)  
#            self.parameters += [kernel, biases]
            self.parameters.append([kernel, biases])
            
        # conv3_2_1
        with tf.name_scope('conv3_2_1') as scope:
            
            kernel = tf.Variable(tf.truncated_normal([3, 3, model_info['conv3_1_2'], model_info['conv3_2_1']], dtype=tf.float32,
                                                     stddev=1e-2), name='weights')
            conv = tf.nn.conv2d(self.layer10_input, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[model_info['conv3_2_1']], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
#            regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
#            out = tf.layers.conv2d(self.conv3_1_2,model_info['conv3_2_1'],3,kernel_regularizer=regularizer,strides=(1, 1), padding='same')
            
            # Batch Normalize
            fc_mean, fc_var = tf.nn.moments(
                out,
                axes=[0,1,2],   # the dimension you wanna normalize, here [0] for batch
                            # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
            )
            scale = tf.Variable(tf.ones([model_info['conv3_2_1']]), name='norms/scale')
            shift = tf.Variable(tf.zeros([model_info['conv3_2_1']]), name='norms/shift')
            epsilon = 0.001
            # apply moving average for mean and var when train on batch
            ema = tf.train.ExponentialMovingAverage(decay=0.5)
            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean), tf.identity(fc_var)
            mean, var = mean_var_with_update()
            
#            out = BatchNormalization()(out)
            out = tf.nn.batch_normalization(out, mean, var, shift, scale, epsilon)
            #-------------------------------------------------------------------
            
            
            self.conv3_2_1 = tf.nn.relu(out, name=scope)
#            print(self.conv3_2_1.get_shape())
#            self.conv3_2_1 = tf.nn.dropout(self.conv3_2_1, self.keep_prob)  
#            self.parameters += [kernel, biases]
            self.parameters.append([kernel, biases])
            
        # conv3_2_2
        with tf.name_scope('conv3_2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, model_info['conv3_2_1'], model_info['conv3_2_2']], dtype=tf.float32,
                                                     stddev=1e-2), name='weights')
            conv = tf.nn.conv2d(self.conv3_2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[model_info['conv3_2_2']], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
#            regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
#            out = tf.layers.conv2d(self.conv3_2_1,model_info['conv3_2_2'],3,kernel_regularizer=regularizer,strides=(1, 1), padding='same')
            
            # Batch Normalize
            fc_mean, fc_var = tf.nn.moments(
                out,
                axes=[0,1,2],   # the dimension you wanna normalize, here [0] for batch
                            # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
            )
            scale = tf.Variable(tf.ones([model_info['conv3_2_2']]), name='norms/scale')
            shift = tf.Variable(tf.zeros([model_info['conv3_2_2']]), name='norms/shift')
            epsilon = 0.001
            # apply moving average for mean and var when train on batch
            ema = tf.train.ExponentialMovingAverage(decay=0.5)
            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean), tf.identity(fc_var)
            mean, var = mean_var_with_update()
            
#            out = BatchNormalization()(out)
            out = tf.nn.batch_normalization(out, mean, var, shift, scale, epsilon)
            #-------------------------------------------------------------------
           
#            out = tf.add(self.resize_conv3_1_2,out)
            
            self.conv3_2_2 = out
            self.layer12_input = tf.add(self.layer10_input,self.conv3_2_2)
            self.layer12_input = tf.nn.relu(self.layer12_input, name=scope)
            
#            print(self.conv3_2_2.get_shape())
#            self.resize_conv3_2_2 = self.feature_map_to_next_layer(self.conv3_2_2,"conv3_3_2")
#            self.conv3_2_2 = tf.nn.dropout(self.conv3_2_2, self.keep_prob)  
#            self.parameters += [kernel, biases]
            self.parameters.append([kernel, biases])
            
        # conv3_3_1
        with tf.name_scope('conv3_3_1') as scope:           
            kernel = tf.Variable(tf.truncated_normal([3, 3, model_info['conv3_2_2'], model_info['conv3_3_1']], dtype=tf.float32,
                                                     stddev=1e-2), name='weights')
            conv = tf.nn.conv2d(self.layer12_input, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[model_info['conv3_3_1']], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
#            regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
#            out = tf.layers.conv2d(self.conv3_2_2,model_info['conv3_3_1'],3,kernel_regularizer=regularizer,strides=(1, 1), padding='same')
            
            # Batch Normalize
            fc_mean, fc_var = tf.nn.moments(
                out,
                axes=[0,1,2],   # the dimension you wanna normalize, here [0] for batch
                            # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
            )
            scale = tf.Variable(tf.ones([model_info['conv3_3_1']]), name='norms/scale')
            shift = tf.Variable(tf.zeros([model_info['conv3_3_1']]), name='norms/shift')
            epsilon = 0.001
            # apply moving average for mean and var when train on batch
            ema = tf.train.ExponentialMovingAverage(decay=0.5)
            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean), tf.identity(fc_var)
            mean, var = mean_var_with_update()
            
#            out = BatchNormalization()(out)
            out = tf.nn.batch_normalization(out, mean, var, shift, scale, epsilon)
            #-------------------------------------------------------------------
            
            self.conv3_3_1 = tf.nn.relu(out, name=scope)
#            self.conv3_3_1 = tf.nn.dropout(self.conv3_3_1, self.keep_prob)  
#            self.parameters += [kernel, biases]
            self.parameters.append([kernel, biases])
            
            # conv3_3_2
        with tf.name_scope('conv3_3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, model_info['conv3_3_1'], model_info['conv3_3_2']], dtype=tf.float32,
                                                     stddev=1e-2), name='weights')
            conv = tf.nn.conv2d(self.conv3_3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[model_info['conv3_3_2']], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
#            regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
#            out = tf.layers.conv2d(self.conv3_3_1,model_info['conv3_3_2'],3,kernel_regularizer=regularizer,strides=(1, 1), padding='same')
            
            # Batch Normalize
            fc_mean, fc_var = tf.nn.moments(
                out,
                axes=[0,1,2],   # the dimension you wanna normalize, here [0] for batch
                            # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
            )
            scale = tf.Variable(tf.ones([model_info['conv3_3_2']]), name='norms/scale')
            shift = tf.Variable(tf.zeros([model_info['conv3_3_2']]), name='norms/shift')
            epsilon = 0.001
            # apply moving average for mean and var when train on batch
            ema = tf.train.ExponentialMovingAverage(decay=0.5)
            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean), tf.identity(fc_var)
            mean, var = mean_var_with_update()
            
#            out = BatchNormalization()(out)
            out = tf.nn.batch_normalization(out, mean, var, shift, scale, epsilon)
            #-------------------------------------------------------------------
            self.conv3_3_2 = out
            self.layer14_input = tf.add(self.layer12_input,self.conv3_3_2)
            self.layer14_input = tf.nn.relu(self.layer14_input, name=scope)
            

#            self.resize_conv3_3_2 = tf.layers.conv2d(self.conv3_3_2,model_info['conv4_1_2'],1,kernel_regularizer=regularizer,strides=(2, 2), padding='same')
#            self.resize_conv3_3_2 = self.feature_map_to_next_layer(self.resize_conv3_3_2,"conv4_1_2")
#            self.conv3_3_2 = tf.nn.dropout(self.conv3_3_2, self.keep_prob)  
#            self.parameters += [kernel, biases]
            self.parameters.append([kernel, biases])
            
#            out = tf.add(self.resize_conv3_2_2,out)          
            

        
        # conv4_1_1
        with tf.name_scope('conv4_1_1') as scope:            
            kernel = tf.Variable(tf.truncated_normal([3, 3, model_info['conv3_3_2'], model_info['conv4_1_1']], dtype=tf.float32,
                                                     stddev=1e-2), name='weights')
            conv = tf.nn.conv2d(self.layer14_input, kernel, [1, 2, 2, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[model_info['conv4_1_1']], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
#            regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
#            out = tf.layers.conv2d(self.conv3_3_2,model_info['conv4_1_1'],3,kernel_regularizer=regularizer,strides=(2, 2), padding='same')
            
            # Batch Normalize
            fc_mean, fc_var = tf.nn.moments(
                out,
                axes=[0,1,2],   # the dimension you wanna normalize, here [0] for batch
                            # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
            )
            scale = tf.Variable(tf.ones([model_info['conv4_1_1']]), name='norms/scale')
            shift = tf.Variable(tf.zeros([model_info['conv4_1_1']]), name='norms/shift')
            epsilon = 0.001
            # apply moving average for mean and var when train on batch
            ema = tf.train.ExponentialMovingAverage(decay=0.5)
            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean), tf.identity(fc_var)
            mean, var = mean_var_with_update()
            
#            out = BatchNormalization()(out)
            out = tf.nn.batch_normalization(out, mean, var, shift, scale, epsilon)
            #-------------------------------------------------------------------
            
            self.conv4_1_1 = tf.nn.relu(out, name=scope)
#            self.conv4_1_1 = tf.nn.dropout(self.conv4_1_1, self.keep_prob)  
#            self.parameters += [kernel, biases]
            self.parameters.append([kernel, biases])

# =============================================================================
#         # pool1
#         self.pool1 = tf.nn.max_pool(self.conv1_2,
#                                ksize=[1, 2, 2, 1],
#                                strides=[1, 2, 2, 1],
#                                padding='SAME',
#                                name='pool1')
# =============================================================================
        
        # conv4_1_2
        with tf.name_scope('conv4_1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, model_info['conv4_1_1'], model_info['conv4_1_2']], dtype=tf.float32,
                                                     stddev=1e-2), name='weights')
            conv = tf.nn.conv2d(self.conv4_1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[model_info['conv4_1_2']], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
#            regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
#            out = tf.layers.conv2d(self.conv4_1_1,model_info['conv4_1_2'],3,kernel_regularizer=regularizer,strides=(1, 1), padding='same')
            
            # Batch Normalize
            fc_mean, fc_var = tf.nn.moments(
                out,
                axes=[0,1,2],   # the dimension you wanna normalize, here [0] for batch
                            # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
            )
            scale = tf.Variable(tf.ones([model_info['conv4_1_2']]), name='norms/scale')
            shift = tf.Variable(tf.zeros([model_info['conv4_1_2']]), name='norms/shift')
            epsilon = 0.001
            # apply moving average for mean and var when train on batch
            ema = tf.train.ExponentialMovingAverage(decay=0.5)
            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean), tf.identity(fc_var)
            mean, var = mean_var_with_update()
            
#            out = BatchNormalization()(out)
            out = tf.nn.batch_normalization(out, mean, var, shift, scale, epsilon)
            #-------------------------------------------------------------------
            
#            out = tf.add(self.resize_conv3_3_2,out)
            
            self.conv4_1_2 = out
            kernel_resize = tf.Variable(tf.truncated_normal([1, 1, model_info['conv3_3_2'], model_info['conv4_1_2']], dtype=tf.float32,
                                                     stddev=1e-2), name='weights')
            self.layer14_input_resize = tf.nn.conv2d(self.layer14_input, kernel_resize, [1, 2, 2, 1], padding='SAME')
            self.layer16_input = tf.add(self.layer14_input_resize,self.conv4_1_2)
            self.layer16_input = tf.nn.relu(self.layer16_input, name=scope)
#            self.resize_conv4_1_2 = self.feature_map_to_next_layer(self.conv4_1_2,"conv4_2_2")
#            self.conv4_1_2 = tf.nn.dropout(self.conv4_1_2, self.keep_prob)  
#            self.parameters += [kernel, biases]
            self.parameters.append([kernel, biases])
            
        # conv4_2_1
        with tf.name_scope('conv4_2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, model_info['conv4_1_2'], model_info['conv4_2_1']], dtype=tf.float32,
                                                     stddev=1e-2), name='weights')
            conv = tf.nn.conv2d(self.layer16_input, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[model_info['conv4_2_1']], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
#            regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
#            out = tf.layers.conv2d(self.conv4_1_2,model_info['conv4_2_1'],3,kernel_regularizer=regularizer,strides=(1, 1), padding='same')
            
            # Batch Normalize
            fc_mean, fc_var = tf.nn.moments(
                out,
                axes=[0,1,2],   # the dimension you wanna normalize, here [0] for batch
                            # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
            )
            scale = tf.Variable(tf.ones([model_info['conv4_2_1']]), name='norms/scale')
            shift = tf.Variable(tf.zeros([model_info['conv4_2_1']]), name='norms/shift')
            epsilon = 0.001
            # apply moving average for mean and var when train on batch
            ema = tf.train.ExponentialMovingAverage(decay=0.5)
            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean), tf.identity(fc_var)
            mean, var = mean_var_with_update()
            
#            out = BatchNormalization()(out)
            out = tf.nn.batch_normalization(out, mean, var, shift, scale, epsilon)
            #-------------------------------------------------------------------
            
            self.conv4_2_1 = tf.nn.relu(out, name=scope)
#            self.conv4_2_1 = tf.nn.dropout(self.conv4_2_1, self.keep_prob)  
#            self.parameters += [kernel, biases]
            self.parameters.append([kernel, biases])
            
        # conv4_2_2
        with tf.name_scope('conv4_2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, model_info['conv4_2_1'], model_info['conv4_2_2']], dtype=tf.float32,
                                                     stddev=1e-2), name='weights')
            conv = tf.nn.conv2d(self.conv4_2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[model_info['conv4_2_2']], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
#            regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
#            out = tf.layers.conv2d(self.conv4_2_1,model_info['conv4_2_2'],3,kernel_regularizer=regularizer,strides=(1, 1), padding='same')
            
            # Batch Normalize
            fc_mean, fc_var = tf.nn.moments(
                out,
                axes=[0,1,2],   # the dimension you wanna normalize, here [0] for batch
                            # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
            )
            scale = tf.Variable(tf.ones([model_info['conv4_2_2']]), name='norms/scale')
            shift = tf.Variable(tf.zeros([model_info['conv4_2_2']]), name='norms/shift')
            epsilon = 0.001
            # apply moving average for mean and var when train on batch
            ema = tf.train.ExponentialMovingAverage(decay=0.5)
            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean), tf.identity(fc_var)
            mean, var = mean_var_with_update()
            
#            out = BatchNormalization()(out)
            out = tf.nn.batch_normalization(out, mean, var, shift, scale, epsilon)
            #-------------------------------------------------------------------
            
#            out = tf.add(self.resize_conv4_1_2,out)
            
            self.conv4_2_2 = out
            self.layer18_input = tf.add(self.layer16_input,self.conv4_2_2)
            self.layer18_input = tf.nn.relu(self.layer18_input, name=scope)
#            self.resize_conv4_2_2 = self.feature_map_to_next_layer(self.conv4_2_2,"conv4_3_2")
                        
#            self.conv4_2_2 = tf.nn.dropout(self.conv4_2_2, self.keep_prob)  
#            self.parameters += [kernel, biases]
            self.parameters.append([kernel, biases])
            
        # conv4_3_1
        with tf.name_scope('conv4_3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, model_info['conv4_2_2'], model_info['conv4_3_1']], dtype=tf.float32,
                                                     stddev=1e-2), name='weights')
            conv = tf.nn.conv2d(self.layer18_input, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[model_info['conv4_3_1']], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
#            regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
#            out = tf.layers.conv2d(self.conv4_2_2,model_info['conv4_3_1'],3,kernel_regularizer=regularizer,strides=(1, 1), padding='same')
            
            # Batch Normalize
            fc_mean, fc_var = tf.nn.moments(
                out,
                axes=[0,1,2],   # the dimension you wanna normalize, here [0] for batch
                            # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
            )
            scale = tf.Variable(tf.ones([model_info['conv4_3_1']]), name='norms/scale')
            shift = tf.Variable(tf.zeros([model_info['conv4_3_1']]), name='norms/shift')
            epsilon = 0.001
            # apply moving average for mean and var when train on batch
            ema = tf.train.ExponentialMovingAverage(decay=0.5)
            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean), tf.identity(fc_var)
            mean, var = mean_var_with_update()
            
#            out = BatchNormalization()(out)
            out = tf.nn.batch_normalization(out, mean, var, shift, scale, epsilon)
            #-------------------------------------------------------------------
            
            self.conv4_3_1 = tf.nn.relu(out, name=scope)
#            self.conv4_3_1 = tf.nn.dropout(self.conv4_3_1, self.keep_prob)  
#            self.parameters += [kernel, biases]
            self.parameters.append([kernel, biases])
        # conv4_3_2
        with tf.name_scope('conv4_3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, model_info['conv4_3_1'], model_info['conv4_3_2']], dtype=tf.float32,
                                                     stddev=1e-2), name='weights')
            conv = tf.nn.conv2d(self.conv4_3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[model_info['conv4_3_2']], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
#            regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
#            out = tf.layers.conv2d(self.conv4_3_1,model_info['conv4_3_2'],3,kernel_regularizer=regularizer,strides=(1, 1), padding='same')
            
            # Batch Normalize
            fc_mean, fc_var = tf.nn.moments(
                out,
                axes=[0,1,2],   # the dimension you wanna normalize, here [0] for batch
                            # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
            )
            scale = tf.Variable(tf.ones([model_info['conv4_3_2']]), name='norms/scale')
            shift = tf.Variable(tf.zeros([model_info['conv4_3_2']]), name='norms/shift')
            epsilon = 0.001
            # apply moving average for mean and var when train on batch
            ema = tf.train.ExponentialMovingAverage(decay=0.5)
            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean), tf.identity(fc_var)
            mean, var = mean_var_with_update()
            
#            out = BatchNormalization()(out)
            out = tf.nn.batch_normalization(out, mean, var, shift, scale, epsilon)
            #-------------------------------------------------------------------
            
#            out = tf.add(self.resize_conv4_2_2,out)
            
            self.conv4_3_2 = out
            self.layer20_input = tf.add(self.layer18_input,self.conv4_3_2)
            self.layer20_input = tf.nn.relu(self.layer20_input, name=scope)
#            self.conv4_3_2 = tf.nn.dropout(self.conv4_3_2, self.keep_prob)  
#            self.parameters += [kernel, biases]
            self.parameters.append([kernel, biases])
            

        
        # fc1
        with tf.name_scope('fc1') as scope:            
            #average_pool
            self.pool = tf.nn.avg_pool(self.layer20_input,
#                               ksize=[1, self.conv4_3_2.get_shape()[0], self.conv4_3_2.get_shape()[1], 1],
                               ksize=[1, 8, 8, 1],
                               strides=[1, 8, 8, 1],
                               padding='SAME',
                               name='pool')
            
            shape = int(np.prod(self.pool.get_shape()[1:]))
            fc1w = tf.Variable(tf.truncated_normal([shape, model_info['fc1']],
                                                         dtype=tf.float32,
                                                         stddev=1e-2), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[model_info['fc1']], dtype=tf.float32),
                                 trainable=True, name='biases')
            output_flat = tf.reshape(self.pool, [-1, shape])
#            output_flat = tf.nn.dropout(output_flat, self.keep_prob)
            
            
            
            self.fc1 = tf.nn.bias_add(tf.matmul(output_flat, fc1w), fc1b)
            
            # Batch Normalize
            fc_mean, fc_var = tf.nn.moments(
                self.fc1,
                axes=[0],   # the dimension you wanna normalize, here [0] for batch
                            # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
            )
            scale = tf.Variable(tf.ones([model_info['fc1']]), name='norms/scale')
            shift = tf.Variable(tf.zeros([model_info['fc1']]), name='norms/shift')
            epsilon = 0.001
            # apply moving average for mean and var when train on batch
            ema = tf.train.ExponentialMovingAverage(decay=0.5)
            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean), tf.identity(fc_var)
            mean, var = mean_var_with_update()
            
            self.probs = tf.nn.batch_normalization(self.fc1, mean, var, shift, scale, epsilon)
#            self.fc1 = BatchNormalization()(self.fc1)
            #-------------------------------------------------------------------
            
            
#            self.fc1 = tf.nn.relu(fc1l)
#            self.parameters += [fc1w, fc1b]
            self.parameters.append([fc1w, fc1b])
            
#            self.fc1_drop = tf.nn.dropout(self.fc1, self.keep_prob)

# =============================================================================
#         # fc2
#         with tf.name_scope('fc2') as scope:
#             self.fc3w = tf.Variable(tf.truncated_normal([model_info['fc1'], model_info['fc2']],
#                                                          dtype=tf.float32,
#                                                          stddev=1e-2), name='weights')
#             self.fc3b = tf.Variable(tf.constant(1.0, shape=[model_info['fc2']], dtype=tf.float32),
#                                  trainable=True, name='biases')
#             self.fc3l = tf.nn.bias_add(tf.matmul(self.fc1_drop, self.fc3w), self.fc3b)
#             self.parameters += [self.fc3w, self.fc3b]
#         
# =============================================================================
        
        # softmax
#        self.probs = tf.nn.softmax(self.fc1)
        
        self.cla_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.probs, labels=self.labs))
#        self.reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.reg_loss = tf.constant(0.0, tf.float32)
        self.reg_constant = 0.0001  # Choose an appropriate one.
        self.regularizer = tf.keras.regularizers.l2(self.reg_constant)
        for weight in tf.trainable_variables():
                if 'weights:' in weight.name or 'biases:' in weight.name:
                    self.reg_loss = tf.add(self.reg_loss,self.regularizer(weight))
        self.loss = tf.add(self.cla_loss, self.reg_loss)
        self.loss = self.cla_loss
        
#        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
#        self.train_step = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,momentum=0.9, use_nesterov=True).minimize(self.loss)
        self.train_step = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,momentum=0.9, use_nesterov=True).minimize(self.loss)
        
        #self.enableDataset()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        #self.printTrainableVariables()
        
        #self.model_variables = self.getVariables()
        
        
# =============================================================================
#     def train(self, images_data, labels_data, epoch = 5, batch_size = 50):
#         self.active="train"
#         print(self.lr)
#         for i in range(epoch):
# #            print("\r訓練進度：||||| - 訓練中 "  + str(i) + "step", end=' ')
#             #image_data, label_data = self.sess.run([self.train_image_batch, self.train_label_batch])
#             
#             total_batch = int(images_data.shape[0] / batch_size)
#             loss_count = 0
#             pred_count = 0
# 
#             for step in range(total_batch):
# #                print("666")
#                 print("\r" + str(step), end=' ')
#                 image_data = images_data[step * batch_size : step * batch_size + batch_size]
#                 #image_data = random_crop_and_flip(image_data)
#                 image_data = random_flip(image_data)
#                 #image_data = whitening_image(image_data)
#                 #image_data = self.augmentData(image_data)
#                 label_data = labels_data[step * batch_size : step * batch_size + batch_size]
#                 
#                 _, loss, pred = self.sess.run([self.train_step, self.loss, self.probs], feed_dict={self.imgs: image_data, self.labs: label_data, self.keep_prob: 0.5, self.keep_prob_conv:0.5, self.keep_prob_conv_1:0.3})
#                 loss_count += loss
#                 
#                 #top1
#                 ans = np.argmax(pred, axis=1)
#                 truth = np.argmax(label_data,axis=1)
#                 ans = np.equal(truth,ans)
#                 #bool轉float
#                 #ans = np.cast(ans,np.float32)
#                 ans = np.float64(ans)
#                 #計算平均
#                 ans = np.mean(ans)
#                 pred_count += ans
#             
#             print('loss : ',loss_count/total_batch)
#             print('pred : ',pred_count/total_batch)
# =============================================================================
 
    def train(self, images_data, labels_data, epoch = 5, batch_size = 50):            
        loss_count = 0
        reg_count = 0
        pred_count = 0
        for i in range(epoch):
            for i in range(len(images_data)):
                print("\r" + str(i), end=' ')
                image_data = images_data[i]
                label_data = labels_data[i]
                
                self.parameters = []
                reg_count += self.sess.run(self.reg_loss)
                _, loss, pred = self.sess.run([self.train_step, self.loss, self.probs], feed_dict={self.imgs: image_data, self.labs: label_data, self.keep_prob: 0.5, self.keep_prob_conv:0.5, self.keep_prob_conv_1:0.3})                
                loss_count += loss
                
                #top1
                ans = np.argmax(pred, axis=1)
                truth = np.argmax(label_data,axis=1)
                ans = np.equal(truth,ans)
                #bool轉float
                #ans = np.cast(ans,np.float32)
                ans = np.float64(ans)
                #計算平均
                ans = np.mean(ans)
                pred_count += ans
            
            print('loss : ',loss_count/len(images_data))
            print('pred : ',pred_count/len(images_data))
            print('regularization loss : ',reg_count/len(images_data))
        return pred_count/len(images_data)
            
            
    def test(self, images_data, labels_data, batch_size = 5000):
        total_batch = int(images_data.shape[0] / batch_size)
        top1_count = 0
        
        
        for step in range(total_batch):
            print("\r"+ "Test Step：" + str(step+1) + "/" + str(total_batch), end=' ')
            image_data = images_data[step * batch_size : step * batch_size + batch_size]
            label_data = labels_data[step * batch_size : step * batch_size + batch_size]
            
            pred = self.sess.run(self.probs, feed_dict={self.imgs: image_data, self.keep_prob: 1.0,self.keep_prob_conv:1.0, self.keep_prob_conv_1:1.0})
            
            #top1
            ans = np.argmax(pred, axis=1)
            truth = np.argmax(label_data,axis=1)
            ans = np.equal(truth,ans)
            #bool轉float
            #ans = np.cast(ans,np.float32)
            ans = np.float64(ans)
            #計算平均
            ans = np.mean(ans)
            top1_count += ans
        print("")

            
            
        top1_accuracy = (top1_count/total_batch)*100
        
        return top1_accuracy
    
    def printTrainableVariables(self):
        for var in tf.trainable_variables():
            print(var.name + str(var.shape))
            
    def writeInfoToTxt(self, path=r'.\log.txt'):
        with open(path, 'a') as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            f.write("\n")
            for var in tf.trainable_variables():
                f.write(var.name + str(var.shape))
                f.write("\n")
                
    def getVariables(self):
        var_list = {}
        for var in tf.trainable_variables():
            var_np = self.sess.run(var)
            var_list[var.name] = var_np
        return var_list