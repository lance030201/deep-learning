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

class AlexNet():
    def __init__(self):
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    
        self.sess = tf.Session()
        
        self.model_info = { "conv1" : 96,
                            "conv2" : 256,
                            "conv3" : 384,
                            "conv4" : 384,
                            "conv5" : 256,
                            "fc1" : 4096,
                            "fc2" : 4096,
                            "fc3" : 10,}
        self.model_variables = {}
        
        self.norm = True
        
        self.model_affect = {"conv1" : "conv2",
                             "conv2" : "conv3",
                             "conv3" : "conv4",
                             "conv4" : "conv5",
                             "conv5" : "fc1",
                             "fc1" : "fc2",
                             "fc2" : "fc3"}
        
        
        
    def reset(self):
        tf.reset_default_graph()
        self.sess = tf.Session()
    
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
            print('保存文件失敗，請設置好路徑，確認路徑存在')
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
        
        self.learning_rate = tf.Variable(float(0), trainable=False, dtype=tf.float32)  
        
        if(model_info == None):
            model_info = self.model_info.copy()
        else :
            self.model_info = model_info.copy()
            
        
        self.sess = tf.Session()
        
        
        self.augmentation_in = tf.placeholder(tf.float32, [None, 32, 32, 3])
        self.augmentation_in = tf.image.random_flip_up_down(self.augmentation_in)
        self.augmentation_in = tf.image.random_flip_left_right(self.augmentation_in)
        
        self.augmentation_in = tf.image.resize_image_with_crop_or_pad(self.augmentation_in,36,36)
        
        #self.augmentation_in= tf.reshape(self.augmentation_in, [36, 36, 3, None])
#        self.augmentation_in = tf.image.random_crop(self.augmentation_in,[100,32,32,3])
        
        self.augmentation_in = tf.image.random_brightness(self.augmentation_in, max_delta=0.05)
        self.augmentation_out = tf.image.random_contrast(self.augmentation_in, 0.8, 1.2)
        self.augmentation_out = tf.image.random_hue(self.augmentation_in, 0.08)
        #self.augmentation_out = tf.image.random_saturation(self.augmentation_in, 0.6, 1.4)
        
        

    
        #紀錄現在的值,(?, 2)
        self.parameters = []
        
        
        self.imgs = tf.placeholder(tf.float32, [None, 32, 32, 3])
        self.labs = tf.placeholder(tf.float32, [None, self.model_info["fc3"]])
        self.keep_prob_conv_1 = tf.placeholder(tf.float32)
        self.keep_prob_conv = tf.placeholder(tf.float32)
        self.keep_prob = tf.placeholder(tf.float32)
        
        
        # conv1
        with tf.name_scope('conv1') as scope:
            kernel = tf.Variable(tf.truncated_normal([11, 11, 3, model_info['conv1']], dtype=tf.float32,
                                                     stddev=1e-2), name='weights')
            conv = tf.nn.conv2d(self.imgs, kernel, [1, 4, 4, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[model_info['conv1']], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
                        
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
            
            out = tf.nn.batch_normalization(out, mean, var, shift, scale, epsilon)
            #-------------------------------------------------------------------
            
            self.conv1_1 = tf.nn.relu(out, name=scope)
            
            self.conv1_1 = tf.nn.dropout(self.conv1_1, self.keep_prob_conv_1)
            
            
            
            
            self.parameters += [kernel, biases]

        # conv2
        with tf.name_scope('conv2') as scope:
            kernel = tf.Variable(tf.truncated_normal([5, 5, model_info['conv1'], model_info['conv2']], dtype=tf.float32,
                                                     stddev=1e-2), name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[model_info['conv2']], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            
            # Batch Normalize
            fc_mean, fc_var = tf.nn.moments(
                out,
                axes=[0,1,2],   # the dimension you wanna normalize, here [0] for batch
                            # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
            )
            scale = tf.Variable(tf.ones([model_info['conv2']]), name='norms/scale')
            shift = tf.Variable(tf.zeros([model_info['conv2']]), name='norms/shift')
            epsilon = 0.001
            # apply moving average for mean and var when train on batch
            ema = tf.train.ExponentialMovingAverage(decay=0.5)
            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean), tf.identity(fc_var)
            mean, var = mean_var_with_update()
            
            out = tf.nn.batch_normalization(out, mean, var, shift, scale, epsilon)
            #-------------------------------------------------------------------
            
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3
        with tf.name_scope('conv3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, model_info['conv2'], model_info['conv3']], dtype=tf.float32,
                                                     stddev=1e-2), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[model_info['conv3']], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            
            # Batch Normalize
            fc_mean, fc_var = tf.nn.moments(
                out,
                axes=[0,1,2],   # the dimension you wanna normalize, here [0] for batch
                            # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
            )
            scale = tf.Variable(tf.ones([model_info['conv3']]), name='norms/scale')
            shift = tf.Variable(tf.zeros([model_info['conv3']]), name='norms/shift')
            epsilon = 0.001
            # apply moving average for mean and var when train on batch
            ema = tf.train.ExponentialMovingAverage(decay=0.5)
            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean), tf.identity(fc_var)
            mean, var = mean_var_with_update()
            
            out = tf.nn.batch_normalization(out, mean, var, shift, scale, epsilon)
            #-------------------------------------------------------------------
            
            self.conv2_1 = tf.nn.relu(out, name=scope)
            
            self.conv2_1 = tf.nn.dropout(self.conv2_1, self.keep_prob_conv)
            
            self.parameters += [kernel, biases]

        # conv4
        with tf.name_scope('conv4') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, model_info['conv3'], model_info['conv4']], dtype=tf.float32,
                                                     stddev=1e-2), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[model_info['conv4']], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            
            # Batch Normalize
            fc_mean, fc_var = tf.nn.moments(
                out,
                axes=[0,1,2],   # the dimension you wanna normalize, here [0] for batch
                            # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
            )
            scale = tf.Variable(tf.ones([model_info['conv4']]), name='norms/scale')
            shift = tf.Variable(tf.zeros([model_info['conv4']]), name='norms/shift')
            epsilon = 0.001
            # apply moving average for mean and var when train on batch
            ema = tf.train.ExponentialMovingAverage(decay=0.5)
            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean), tf.identity(fc_var)
            mean, var = mean_var_with_update()
            
            out = tf.nn.batch_normalization(out, mean, var, shift, scale, epsilon)
            #-------------------------------------------------------------------
            
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]
        # conv5
        with tf.name_scope('conv5') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, model_info['conv4'], model_info['conv5']], dtype=tf.float32,
                                                     stddev=1e-2), name='weights')
            conv = tf.nn.conv2d(self.conv2_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[model_info['conv5']], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            
            # Batch Normalize
            fc_mean, fc_var = tf.nn.moments(
                out,
                axes=[0,1,2],   # the dimension you wanna normalize, here [0] for batch
                            # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
            )
            scale = tf.Variable(tf.ones([model_info['conv5']]), name='norms/scale')
            shift = tf.Variable(tf.zeros([model_info['conv5']]), name='norms/shift')
            epsilon = 0.001
            # apply moving average for mean and var when train on batch
            ema = tf.train.ExponentialMovingAverage(decay=0.5)
            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean), tf.identity(fc_var)
            mean, var = mean_var_with_update()
            
            out = tf.nn.batch_normalization(out, mean, var, shift, scale, epsilon)
            #-------------------------------------------------------------------
            self.conv3_1 = tf.nn.relu(out, name=scope)
            
            self.conv3_1 = tf.nn.dropout(self.conv3_1, self.keep_prob_conv)
            
            self.parameters += [kernel, biases]
            # pool5
            self.pool5 = tf.nn.max_pool(self.conv3_1,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME',
                                   name='pool5')
            
        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            print("#############",shape)
            fc1w = tf.Variable(tf.truncated_normal([shape, model_info['fc1']],
                                                         dtype=tf.float32,
                                                         stddev=1e-2), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[model_info['fc1']], dtype=tf.float32),
                                 trainable=True, name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            
            # Batch Normalize
            fc_mean, fc_var = tf.nn.moments(
                fc1l,
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
            
            fc1l = tf.nn.batch_normalization(fc1l, mean, var, shift, scale, epsilon)
            #-------------------------------------------------------------------
            
            
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]
            
            self.fc1_drop = tf.nn.dropout(self.fc1, self.keep_prob)
            
            
        # fc2
        with tf.name_scope('fc2') as scope:
            self.fc2w = tf.Variable(tf.truncated_normal([model_info['fc1'], model_info['fc2']],
                                                         dtype=tf.float32,
                                                         stddev=1e-2), name='weights')
            self.fc2b = tf.Variable(tf.constant(1.0, shape=[model_info['fc2']], dtype=tf.float32),
                                 trainable=True, name='biases')
            
            self.fc2l = tf.nn.bias_add(tf.matmul(self.fc1_drop, self.fc2w), self.fc2b)
            
            # Batch Normalize
            fc_mean, fc_var = tf.nn.moments(
                self.fc2l,
                axes=[0],   # the dimension you wanna normalize, here [0] for batch
                            # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
            )
            print("++++++++++++",[model_info['fc2']])
            scale = tf.Variable(tf.ones([model_info['fc2']]), name='norms/scale')
            shift = tf.Variable(tf.zeros([model_info['fc2']]), name='norms/shift')
            epsilon = 0.001
            # apply moving average for mean and var when train on batch
            ema = tf.train.ExponentialMovingAverage(decay=0.5)
            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean), tf.identity(fc_var)
            mean, var = mean_var_with_update()
            
            self.fc2l = tf.nn.batch_normalization(self.fc2l, mean, var, shift, scale, epsilon)
            #-------------------------------------------------------------------
            
            
            self.fc2 = tf.nn.relu(self.fc2l)
            self.parameters += [fc1w, fc1b]
            
            self.fc2_drop = tf.nn.dropout(self.fc2, self.keep_prob)

        # fc3
        with tf.name_scope('fc3') as scope:
            self.fc3w = tf.Variable(tf.truncated_normal([model_info['fc2'], model_info['fc3']],
                                                         dtype=tf.float32,
                                                         stddev=1e-2), name='weights')
            self.fc3b = tf.Variable(tf.constant(1.0, shape=[model_info['fc3']], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2_drop, self.fc3w), self.fc3b)
            self.parameters += [self.fc3w, self.fc3b]
        
        
        # softmax
        self.probs = tf.nn.softmax(self.fc3l)
        
        #self.loss = tf.reduce_mean(-tf.reduce_sum(self.labs * tf.log(self.probs), reduction_indices=[1])) # cross_entropy
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.fc3l, labels=self.labs))
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        
        #self.enableDataset()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        #self.printTrainableVariables()
        
        #self.model_variables = self.getVariables()
        
        
    def train(self, images_data, labels_data, epoch = 5, batch_size = 100):
        for i in range(epoch):
            print('目前learning_rate：',self.sess.run(self.learning_rate))
            #print("\r訓練進度：||||| - 訓練中 "  + str(i) + "step", end=' ')
            #image_data, label_data = self.sess.run([self.train_image_batch, self.train_label_batch])
            
            total_batch = int(images_data.shape[0] / batch_size)
            loss_count = 0
            pred_count = 0

            print("")
            for step in range(total_batch):
                print("\r" + str(step), end=' ')
                image_data = images_data[step * batch_size : step * batch_size + batch_size]
                #image_data = random_crop_and_flip(image_data)
                image_data = random_flip(image_data)
                #image_data = whitening_image(image_data)
                #image_data = self.augmentData(image_data)
                label_data = labels_data[step * batch_size : step * batch_size + batch_size]
                
                _, loss, pred = self.sess.run([self.train_step, self.loss, self.probs], feed_dict={self.imgs: image_data, self.labs: label_data, self.keep_prob: 0.5, self.keep_prob_conv:0.5, self.keep_prob_conv_1:0.5})
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
            
            print("")
            print('loss : ',loss_count/total_batch)
            print('pred : ',pred_count/total_batch)
            
            
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
        