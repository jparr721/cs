import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    '''
    PolicyNet defines the policy neural network class
    to determine optimum moves for the agents in the system
    '''
    model_path = 'model/state'
    input_size = 64

    def __init__(self):
        super(PolicyNet, self).__init__()

        # Deep layered nesting allows for greater granularity
        # on the processing of the input data, with the tradeoff
        # of training time, of course
        self.l1 = nn.Linear(1024, 1024)
        self.l2 = nn.Linear(1024, 1024)
        self.l2a = nn.Linear(1024, 1024)
        self.l2b = nn.Linear(1024, 1024)
        self.l3 = nn.Linear(1024, 512)
        self.l4 = nn.Linear(512, 256)
        self.l5 = nn.Linear(256, 128)
        self.l6 = nn.Linear(128, 64)

        # Action out
        # Actions have many different output dimensions
        self.action_head = nn.Linear(64, 5)

        # Value out
        # Values should have only one, the performance
        self.value_head = nn.Linear(64, 1)

    def get_state_value(self, x):
        x = self.thru_layers(x)
        return self.value_head(x)

    def load_state(self):
        self.load_state_dict(torch.load(PolicyNet.model_path))

    def save_state(self):
        torch.save(self.state_dict(), PolicyNet.model_path)

    def thru_layers(self, x):
        '''
        Applies the rectified linear unit activation
        at each layer

        Parameters
        ----------
        x - The input feature vector
        '''
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l2a(x))
        x = F.relu(self.l2b(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))
        x = F.relu(self.l6(x))
        return x

    def forward(self, x):
        '''
        Forward propagates the layers with a softmax activation
        function

        Parameters
        ----------
        x - Our input feature vector to be propagated
        '''
        x = self.thru_layers(x)
        action_scores = self.action_head(x)
        return F.softmax(action_scores, dim=-1)
