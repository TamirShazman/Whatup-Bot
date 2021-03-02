import torch.nn as nn

"""
This model is the neuralNet. The neural has 3 layers (l1, l2, l3)
and the size of each layers is "input_size"(vector in the size
all the words group), "hidden_size"(set to 8) and 
"num_classes"(number of "feeling") corresponded. 
"""


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        :param x:Input Vector
        :return: Vector in the size of the number of feeling
        that each input is number between 0 to 1, correspond to
        the probably.
        """
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out
