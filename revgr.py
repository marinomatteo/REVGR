import torch
import torch.nn as nn
import copy
from torch.autograd import Function
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class GradientReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        out = grad_output.neg() * ctx.alpha
        return out,None

class DANN(nn.Module):

    def __init__(self, num_category=7, test_or_train=2):
        super(DANN, self).__init__()
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
            nn.Linear(4096, num_category),
        )

        self.domain_classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, test_or_train),
        )

    def update_weigth(self):
        self.domain_classifier[1].weight.data =  copy.deepcopy(self.classiﬁer[1].weight.data)
        self.domain_classifier[1].bias.data = copy.deepcopy(self.classiﬁer[1].bias.data)
        self.domain_classifier[4].weight.data = copy.deepcopy(self.classiﬁer[4].weight.data)
        self.domain_classifier[4].bias.data = copy.deepcopy(self.classiﬁer[4].bias.data)

    def forward(self, x, alpha=None):
        x = self.features(x)
        x = self.avgpool(x)
        features = torch.flatten(x, 1)

        classified = -1

        # If we pass alpha, we can assume we are training the discriminator
        if alpha is not None:
            # gradient reversal layer (backward gradients will be reversed)
            reverse_feature = GradientReverse.apply(features,alpha)
            domain_image = self.domain_classifier(reverse_feature)
            classified = domain_image
        else:
            category_image = self.classifier(features)
            classified = category_image

        return classified


def AlexNetDANN(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = DANN(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        
        # removing unused params
        state_dict.popitem("classifier.6.bias")
        state_dict.popitem("classifier.6.weight") 
        model.load_state_dict(state_dict,strict=False)
        model.update_weigth()

    return model
