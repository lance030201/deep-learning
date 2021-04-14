from model import *
from dataset import *

model_info = {  "conv1" : 20,
                "conv2" : 50,
                "fc1" : 500,
                "fc2" : 10,}

        
        
if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels = load_data()
    
    # Construct
    lenet = LeNet()
    lenet.build(model_info)
    
    # Load model
    lenet.load(r'\best_model.ckpt')
    print(lenet.test(test_images, test_labels))

    

































