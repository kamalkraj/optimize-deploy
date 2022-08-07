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
