import os

import torch
from net import Net

from base import BaseModel

torch.manual_seed(0)

MODEL_FILE = "cifar_net.pth"


class Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.name = "CIFA10_Net"
        self.model = Net()
        path = os.path.join(os.path.dirname(__file__), MODEL_FILE)
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        self.auxiliary_data = False

    def get_model(self):
        return self.model

    def get_sample_input(self):
        return [((torch.randn(self.batch_size, 3, 32, 32),), 0)]
