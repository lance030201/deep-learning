from model import *
from dataset import *
import numpy as np
from time import time
import tensorflow as tf

model_info = {  "conv1" : 20,
                "conv2" : 50,
                "fc1" : 500,
                "fc2" : 10,}
                

if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels = load_data()
    
    # Construct
    lenet = LeNet()      
    lenet.build(model_info)
    lenet.sess.run(tf.assign(lenet.learning_rate,1e-3))
    #vgg.save()
#    vgg.load(r'\model\lenet.ckpt')

    
# =============================================================================
#     #儲存網路架構參數成np檔
#     vgg.build(model_info)
#     vgg.load(r'\model\save_output.ckpt')
#     output = vgg.getVariables()
#     for temp in output:
#         temp_name = ""
#         for n in temp:
#             if n=='/' or n==':':
#                 temp_name+='_'
#             else:
#                 temp_name+=n
#         np.save(temp_name,output[temp])
# =============================================================================

    print("模組準確率：",lenet.test(test_images, test_labels))
    max_accuracy = 0
    epochs = 90
    for index in range(epochs):
        print("======================第{}次訓練======================".format(index+1))  
        if index==epochs/3:
            lenet.sess.run(tf.assign(lenet.learning_rate,1e-4))
        if index==epochs*2/3:
            lenet.sess.run(tf.assign(lenet.learning_rate,1e-5))
        start = time()
        lenet.train(train_images, train_labels,epoch = 1)
        accuracy = lenet.test(test_images, test_labels)
        if accuracy>max_accuracy:
            lenet.save(r'\best_model.ckpt')
            max_accuracy = accuracy
        print("模組準確率：",accuracy,"目前最好準確率",max_accuracy)
        print('單次訓練+測試時間：',time()-start)

