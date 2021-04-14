from model import *
from dataset import *
import numpy as np
from time import time

model_info = { "conv1" : 96,
              "conv2" : 256,
              "conv3" : 384,
              "conv4" : 384,
              "conv5" : 256,
              "fc1" : 4096,
              "fc2" : 4096,
              "fc3" : 10,}
                


if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels = load_data()
    
    # Construct
    alexnet = AlexNet()            
    alexnet.build(model_info)    
#    alexnet.load(r'\model\prune\median2_1.ckpt') 
    
    #儲存網路架構參數成np檔
# =============================================================================
#     alexnet.load(r'\model\origin\save.ckpt')
#     output = alexnet.getVariables()
#     for temp in output:
#         temp_name = ""
#         for n in temp:
#             if n=='/' or n==':':
#                 temp_name+='_'
#             else:
#                 temp_name+=n
#         write = "model_np\\" + temp_name
#         np.save(write,output[temp])
# =============================================================================
    
    print("模組準確率：",alexnet.test(test_images, test_labels))
    
    max_accuracy = 0
    alexnet.sess.run(tf.assign(alexnet.learning_rate,1e-3))
    epochs = 150
    for index in range(epochs):
        print("======================第{}次訓練======================".format(index+1))  
        if index==epochs/3:
            alexnet.sess.run(tf.assign(alexnet.learning_rate,1e-4))
        if index==epochs*2/3:
            alexnet.sess.run(tf.assign(alexnet.learning_rate,1e-5))
        start = time()
        alexnet.train(train_images, train_labels,epoch = 1)
        accuracy = alexnet.test(test_images, test_labels)
        if accuracy>max_accuracy:
            alexnet.save(r'\best_model.ckpt')
            max_accuracy = accuracy
        print("模組準確率：",accuracy,"目前最好準確率",max_accuracy)
        print('單次訓練+測試時間：',time()-start)

        
        


    
    
    
        
    