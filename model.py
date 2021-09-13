import torch
from torch.nn import Sequential, BatchNorm2d, ReLU, MaxPool2d, Linear, Conv2d, Module, Dropout

# Class to handle all things of the neural network modelx.
class Model(Module):

    # Initializes the convolutional neural network.    
    def __init__(self):
        super().__init__()
        self.conv_layer = Sequential(Conv2d(3, 24, 3, 1, 1), BatchNorm2d(24), ReLU(),#3 input channels (red, green, blue), 24 output channels, 3x3 filter
                        Conv2d(24, 24, 3, 1, 1), BatchNorm2d(24), ReLU(), MaxPool2d(2), Dropout(0.2), 
                        Conv2d(24, 32, 3, 1, 1), BatchNorm2d(32), ReLU(),
                        Conv2d(32, 64, 3, 1, 1), BatchNorm2d(64), ReLU(), MaxPool2d(2), Dropout(0.2))
        self.linear_layer = Sequential(
                        Linear(64 * 8 * 8, 128), ReLU(), 
                        Linear(128, 64), ReLU(),
                        Linear(64, 10))
  
    # The forward function computes output Tensors from input Tensors.
    # @input     input tensor of the color image.
    # @return    flattened output tensor
    def forward(self, input):
        output = self.conv_layer(input) # 3 color channels, 32x32 pixel image
        #print(output.shape)
        output = output.view(-1, 64 * 8 * 8) 
        output = self.linear_layer(output)  
        return output

