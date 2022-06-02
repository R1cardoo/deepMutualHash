import torch
import torch.nn.functional as F


def load_model(code_length, device):
    """
    Load CNN model.

    Args
        code_length (int): Hashing code length.

    Returns
        model (torch.nn.Module): CNN model.
    """
    # model = AlexNet(code_length)

    model = BrainNetCNN(code_length).to(device)

    # state_dict = load_state_dict_from_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    #                                       'model_params/')
    # model.alexnet.load_state_dict(state_dict, strict=False)

    return model


class E2EBlock(torch.nn.Module):
    '''E2Eblock.'''

    def __init__(self, in_planes, planes, bias=False):
        super(E2EBlock, self).__init__()
        self.d = 90
        self.cnn1 = torch.nn.Conv2d(in_planes, planes, (1, self.d), bias=bias)
        self.cnn2 = torch.nn.Conv2d(in_planes, planes, (self.d, 1), bias=bias)

    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a] * self.d, 3) + torch.cat([b] * self.d, 2)


# BrainNetCNN Network for fitting Gold-MSI on LSD dataset

# In[3]:


class BrainNetCNN(torch.nn.Module):
    def __init__(self, code_length):
        super(BrainNetCNN, self).__init__()
        self.in_planes = 1
        self.d = 90

        self.e2econv1 = E2EBlock(1, 32, bias=True)
        self.e2econv2 = E2EBlock(32, 90, bias=True)
        self.E2N = torch.nn.Conv2d(90, 1, (1, self.d))
        self.N2G = torch.nn.Conv2d(1, 256, (self.d, 1))
        self.dense1 = torch.nn.Linear(256, 128)
        self.dense2 = torch.nn.Linear(128, 30)
        self.dense3 = torch.nn.Linear(30, code_length)  # 最后一层输出哈希的长度

    def forward(self, x):
        out = F.leaky_relu(self.e2econv1(x), negative_slope=0.33)
        out = F.leaky_relu(self.e2econv2(out), negative_slope=0.33)
        out = F.leaky_relu(self.E2N(out), negative_slope=0.33)
        out = F.dropout(F.leaky_relu(self.N2G(out), negative_slope=0.33), p=0.5)
        out = out.view(out.size(0), -1)
        out = F.dropout(F.leaky_relu(self.dense1(out), negative_slope=0.33), p=0.5)
        out = F.dropout(F.leaky_relu(self.dense2(out), negative_slope=0.33), p=0.5)
        out = F.leaky_relu(self.dense3(out), negative_slope=0.33)

        return out
