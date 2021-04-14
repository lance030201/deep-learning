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
#    resnet.load(r"\model\temp\prune_50_25.ckpt")
    
    
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

    max_accuracy = resnet.test(test_images, test_labels)
    print("模組準確率：",max_accuracy)               






