import torch
import torch.nn as nn 
import torchvision.models as models

from .registry import register_model



class ResNet_Encoder(nn.Module):
    def __init__(self, base_model="resnet50", cifar_stem=True, channels_in=3, num_classes=0):
        """obtains the ResNet for use as an Encoder, with the last fc layer
        exchanged for an identity

        Args:
            base_model (str, optional): [description]. Defaults to "resnet50".
            cifar_stem (bool, optional): [input resolution of 32x32]. Defaults to True.
            channels_in (int, optional): [description]. Defaults to 3.
            num_classes (int, optional): number of classes (if = 0) purely feature extractor
        """
        super(ResNet_Encoder, self).__init__()
        self.resnet_dict = {
            "resnet18": models.resnet18(pretrained=False),
            "resnet50": models.resnet50(pretrained=False),
        }

        self.resnet = self._get_basemodel(base_model)
        num_ftrs = self.resnet.fc.in_features

        # change 1st convolution to work with inputs [channels_in x 32 x 32]
        if cifar_stem:
            conv1 = nn.Conv2d(
                channels_in, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            nn.init.kaiming_normal_(conv1.weight, mode="fan_out", nonlinearity="relu")
            self.resnet.conv1 = conv1
            self.resnet.maxpool = nn.Identity()
        elif channels_in != 3:
            conv1 = nn.Conv2d(
                channels_in, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            nn.init.kaiming_normal_(conv1.weight, mode="fan_out", nonlinearity="relu")
            self.resnet.conv1 = conv1
        if base_model == "resnet18":
            self.z_dim = 512
        else:
            self.z_dim = 2048

        if num_classes==0:
            self.resnet.fc = nn.Identity()
        else:
            self.resnet.fc = nn.Linear()

    def forward(self, x):
        return self.resnet(x)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise (
                "Invalid model name. Check the config file and pass one of: resnet18 or resnet50"
            )

@register_model
def get_cls_model(config, base_model, num_classes: int = 10, cifar_stem:bool=True, channels_in:int=3,**kwargs) -> ResNet_Encoder:
    return ResNet_Encoder(base_model=base_model, cifar_stem=cifar_stem, channels_in=channels_in, num_classes=num_classes)