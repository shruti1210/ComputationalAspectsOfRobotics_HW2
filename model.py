import torch
import torch.nn as nn
import torch.nn.functional as F

class convolutionBlock(nn.Module):
    def __init__(self, inputChannel, outputChannel):
        super(convolutionBlock, self).__init__()
        self.conv = nn.Conv2d(inputChannel, outputChannel, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, inputs):
        x = self.conv(inputs)
        return self.relu(x)

class encoderBlock(nn.Module):
    def __init__(self, inputChannel, outputChannel):
        super(encoderBlock, self).__init__()
        self.conv = convolutionBlock(inputChannel, outputChannel)
        self.pool = nn.MaxPool2d((2,2))
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p

class decoderBlock(nn.Module):
    def __init__(self, inputChannel, outputChannel):
        super(decoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(inputChannel, outputChannel, kernel_size=2, stride=2, padding=0)
        self.conv = convolutionBlock(outputChannel+outputChannel, outputChannel)
    
    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        
        return x

class MiniUNet(nn.Module):
    # TODO: implement a neural network as described in the handout
    def __init__(self):
        """Initialize the layers of the network as instance variables."""
        super(MiniUNet, self).__init__()
        # TODO
        self.e1 = encoderBlock(3, 16)
        self.e2 = encoderBlock(16, 32)
        self.e3 = encoderBlock(32, 64)
        self.e4 = encoderBlock(64, 128)
        
        self.b = convolutionBlock(128, 256)
        
        self.d1 = decoderBlock(256, 128)
        self.d2 = decoderBlock(128, 64)
        self.d3 = decoderBlock(64, 32)
        self.d4 = decoderBlock(32, 16)
        
        self.output = nn.Conv2d(16, 6, kernel_size=1, padding=0)        

    def forward(self, x):
        """
        In:
            x: Tensor [batchsize, channel, height, width], channel=3 for rgb input
        Out:
            output: Tensor [batchsize, class, height, width], class=number of objects + 1 for background
        Purpose:
            Forward process. Pass the input x through the layers defined in __init__() to get the output.
        """
        # TODO
        
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        
        b = self.b(p4)
        
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        output = self.output(d4)
        return output


if __name__ == '__main__':
    model = MiniUNet()
    input_tensor = torch.zeros([1, 3, 240, 320])
    output = model(input_tensor)
    print("output size:", output.size())
    print(model)