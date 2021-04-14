from model import *
from dataset import *

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
    
    
    # Load model
    alexnet.load(r'\best_model.ckpt')
    print(alexnet.test(test_images, test_labels))
    
    #儲存網路架構參數成np檔
    # alexnet.load(r'\model\origin\origin2.ckpt')
# =============================================================================
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
# 
# =============================================================================
    


































