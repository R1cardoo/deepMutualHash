import torch.nn as nn
import torchvision
from torchinfo import summary
from torch.hub import load_state_dict_from_url


def load_model(code_length, device):
    """
    Load CNN model.

    Args
        code_length (int): Hashing code length.

    Returns
        model (torch.nn.Module): CNN model.
    """
    # model = AlexNet(code_length)

    model = Beginner(code_length).to(device)

    # state_dict = load_state_dict_from_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    #                                       'model_params/')
    # model.alexnet.load_state_dict(state_dict, strict=False)

    return model


class Beginner(nn.Module):
    def __init__(self, code_length):
        super(Beginner, self).__init__()
        resnet = torchvision.models.resnet18(pretrained=True)

        self.backNet = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            *(list(resnet.children())[1:-1])
        )

        self.fc = nn.Linear(512, 12)

    def forward(self, x):
        x = self.backNet(x).reshape(-1, 512)
        x = self.fc(x)
        return x


class AlexNet(nn.Module):

    def __init__(self, code_length):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1000),
        )

        self.classifier = self.classifier[:-1]
        self.hash_layer = nn.Sequential(
            nn.Linear(4096, code_length),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        x = self.hash_layer(x)
        return x
