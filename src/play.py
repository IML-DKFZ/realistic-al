# import os
# import IPython
# import torch
# import torch.nn as nn
# from pl_bolts.models.self_supervised.simclr.simclr_module import SimCLR
# from data.data import TorchVisionDM
# from models.networks.bayesian_resnet import ResNet

# path = "/home/c817h/Documents/logs_cluster/SSL/SimCLR/cifar10/2021-11-11 16:20:56.103061/checkpoints/last.ckpt"

# model = SimCLR.load_from_checkpoint(path, map_location="cpu", strict=False)

# root = os.getenv("DATA_ROOT")
# dm = TorchVisionDM(root, val_split=5000, dataset="cifar10", active=False)
# dm.prepare_data()
# dm.setup()


# def load_from_ssl_checkpoint(model: nn.Module, path):
#     """Loads the parameters from"""
#     model_ssl = SimCLR.load_from_checkpoint(path, map_location="cpu", strict=False)
#     param_names = model.resnet.state_dict().keys()
#     load_dict = dict([(n, model_ssl.encoder.state_dict()[n]) for n in param_names])
#     skip_names = [
#         n for n in model_ssl.encoder.state_dict().keys() if n not in param_names
#     ]
#     msg = model.resnet.load_state_dict(load_dict)
#     print(msg)
#     print(f"Skipped Parameters: {skip_names}")


# train_loader = dm.train_dataloader()
# x, y = next(iter(train_loader))
# model.eval()
# out = model(x)
# model2 = ResNet(base_model="resnet18", num_classes=10)
# message2 = model2.resnet.load_state_dict(model.encoder.state_dict(), strict=False)
# model3 = ResNet(base_model="resnet18", num_classes=10)


# names = model3.resnet.state_dict().keys()
# dict_params = dict([(n, model.encoder.state_dict()[n]) for n in names])
# left_out = [n for n in model.encoder.state_dict().keys() if n not in names]
# message3 = model3.resnet.load_state_dict(dict_params)

# model4 = ResNet(base_model="resnet18", num_classes=10)
# load_from_ssl_checkpoint(model4, path)


# model2.eval()
# model3.eval()
# model4.eval()

# out2 = model2.get_features(x)
# out3 = model3.resnet(x)
# out4 = model4.resnet(x)


# def sum_sqrt(x, y):
#     return torch.sum((x - y) ** 2)


# df2 = sum_sqrt(out, out2)
# df3 = sum_sqrt(out, out3)
# df4 = sum_sqrt(out, out4)
# df23 = sum_sqrt(out2, out3)
# print(df2)
# print(df3)
# print(df4)
# print(df23)


# IPython.embed()
import sys

from loguru import logger
import numpy as np
from utils.log_utils import setup_logger

# trage = logger.add("{}.log".format(__file__.split(".")[0]))

setup_logger()

logger.trace("trace msg")
logger.debug("Debug msg")
logger.info("info")
logger.warning("warning msg")
logger.success("Success message")
logger.critical("Critical message!")


def main_fct():
    out = 1 / 0
    return out


@logger.catch
def main():
    main_fct()


# print(sys.stderr)

if __name__ == "__main__":
    main()
