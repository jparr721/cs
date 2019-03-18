import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim


class Network(nn.Module):
    def __init__(self, im_height, im_width, seed=1):
        super(Network, self).__init__()
        self.seed = seed
        self.dim = im_height * im_width
        self.learning_rate = 0.1

        self.seed = torch.manual_seed(self.seed)
        self.kernel_xy = self.dim / 4
        self.kernel_size = (self.kernel_xy, self.kernel_xy)

        # Construct our 2d convolutional layers
        self.conv1 = nn.Conv2d(self.dim, self.dim, self.kernel_size)
        self.conv2 = nn.Conv2d(self.dim, 2*self.dim, self.kernel_size)

        # Pool our layers
        self.pool1 = nn.AvgPool2d(self.kernel_size)

        # Initiate our recurrent layers
        self.lstm1 = nn.LSTM(self.dim, self.dim)

        # Add our fully connected layers
        self.fc1 = nn.Linear(self.dim, self.dim)

    def forward(self, x):
        x = self.lstm(x, 1, -1)
        scores = F.log_softmax(x)

        return scores
