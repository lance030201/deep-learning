from model import *
from dataset import *
import numpy as np
from time import time

model_info = {  "conv1_1" : 64,
                "conv1_2" : 64,
                "conv2_1" : 128,
                "conv2_2" : 128,
                "conv3_1" : 256,
                "conv3_2" : 256,
                "conv3_3" : 256,
                "conv4_1" : 512,
                "conv4_2" : 512,
                "conv4_3" : 512,
                "conv5_1" : 512,
                "conv5_2" : 512,
                "conv5_3" : 512,
                "fc1" : 512,
                "fc2" : 10,}

    
if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels = load_data()
    
    # Construct
    vgg = Vgg16()
    vgg.build(model_info)
    
    # Load model
#    vgg.load(r'\model\origin\save.ckpt')
    #vgg.save()
    
#儲存網路架構參數成np檔
# =============================================================================
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


    print("模組準確率：",vgg.test(test_images, test_labels))
    
    max_accuracy = 0
    vgg.sess.run(tf.assign(vgg.learning_rate,1e-3))
    epochs = 300
    for index in range(epochs):
        print("======================第{}次訓練======================".format(index+1))  
        if index==epochs/3:
            vgg.sess.run(tf.assign(vgg.learning_rate,1e-4))
        if index==epochs*2/3:
            vgg.sess.run(tf.assign(vgg.learning_rate,1e-5))
        start = time()
        vgg.train(train_images, train_labels,epoch = 1)
        accuracy = vgg.test(test_images, test_labels)
        if accuracy>max_accuracy:
            vgg.save(r'\best_model.ckpt')
            max_accuracy = accuracy
        print("模組準確率：",accuracy,"目前最好準確率",max_accuracy)
        print('單次訓練+測試時間：',time()-start)
        
        
        


