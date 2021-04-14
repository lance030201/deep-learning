from model import *
from dataset import *
import numpy as np

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
#    vgg.save()
    vgg.load(r'\best_model.ckpt')
    print(vgg.test(test_images, test_labels))

    

































