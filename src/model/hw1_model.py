from torch import nn
from torch.nn import Linear, ReLU

class hw1_model(nn.Module) :
    def __init__(self, input_size = 784, num_classes = 10, width = 100, depth = 10) :
        super(hw1_model, self).__init__()
        self.input_size = input_size
        self.depth = depth
        self.linear_layers = nn.ModuleList([Linear(input_size, width)] + [Linear(width, width) for _ in range(depth-1)])
        self.relu_layers = nn.ModuleList([ReLU(inplace=True) for _ in range(depth)])
        self.classifier = Linear(width, num_classes)

    def forward(self, x) :
        x.float()
        x = x.view(-1, self.input_size)
        for i in range(self.depth) :
            x = self.linear_layers[i](x)
            x = self.relu_layers[i](x)
        x = self.classifier(x)
        return x
