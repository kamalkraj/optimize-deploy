import os
import urllib

import torch

from base import BaseModel

torch.manual_seed(0)


class Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.name = "ResNet50_Dynamic"
        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0", "resnet50", pretrained=True
        )
        self.model.eval()
        self.auxiliary_data = True

    def get_model(self):
        return self.model

    def get_sample_input(self):
        return [((torch.randn(self.batch_size, 3, 224, 224),), 0)]

    def get_dynamic_info(self):
        dynamic_info = {
            "inputs": [{0: "batch", 2: "height", 3: "width"}],
            "outputs": [{0: "batch"}],
        }
        return dynamic_info

    def save_auxiliary_data(self, path: str):
        filename = os.path.join(path, "labels.txt")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",  # noqa E501
            filename,
        )
