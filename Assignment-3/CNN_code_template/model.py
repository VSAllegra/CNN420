"""
define moduals of model
"""
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch

class CNNModel(nn.Module):
    
    def __init__(self, args):
        super(CNNModel, self).__init__()
        ##-----------------------------------------------------------
        ## define the model architecture here
        ## MNIST image input size batch * 28 * 28 (one input channel)
        ##-----------------------------------------------------------
        
        ## define CNN layers below
        self.conv_layer_1 = nn.Sequential(
                                    nn.Conv2d(in_channels=1, out_channels=16, kernel_size=args.k_size, stride=args.stride, padding=0),
                                    nn.ReLU(),
                                 )
        self.conv_layer_2 =  nn.Sequential( 
                                    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=args.k_size, stride=args.stride, padding=0),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(32),
                                    nn.MaxPool2d(kernel_size=2),
                                 )
        self.conv_layer_3 =  nn.Sequential( 
                                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=args.k_size, stride=args.stride, padding=0),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(64),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.Dropout(args.dropout)
                                 )
        
        ##------------------------------------------------
        ## write code to define fully connected layer below
        ##------------------------------------------------
        in_size = 64 * 4 * 4
        out_size = 10
        self.fc = nn.Linear(in_size, out_size)

    '''feed features to the model'''
    def forward(self, x):
        ##---------------------------------------------------------
        ## write code to feed input features to the CNN models defined above
        ##---------------------------------------------------------
        #print(x.size())
        x_out = self.conv_layer_1(x)
        #print(x_out.size())
        x_out = self.conv_layer_2(x_out)
        #print(x_out.size())
        x_out = self.conv_layer_3(x_out)
        #print(x_out.size())

        ## write flatten tensor code below (it is done)
        x_out = torch.flatten(x_out,1) # x_out is output of last layer

        ## ---------------------------------------------------
        ## write fully connected layer (Linear layer) below
        ## ---------------------------------------------------
        x_out = self.fc(x_out)
        result = x_out
            
        return result
