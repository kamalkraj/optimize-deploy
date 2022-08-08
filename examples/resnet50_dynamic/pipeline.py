import os

import torch
from torchvision import transforms


class Pipeline:
    def __init__(self, path) -> None:
        super().__init__()
        self.transformer = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        with open(os.path.join(path, "labels.txt"), "r") as f:
            self.categories = [s.strip() for s in f.readlines()]

    def preprocess(self, image: torch.Tensor, device: str) -> torch.Tensor:
        return self.transformer(image).to(device)

    def postprocess(self, output: torch.Tensor, top_k: int = 5) -> dict:
        probabilities = torch.nn.functional.softmax(output, dim=0)
        top_probabilities, top_indices = torch.topk(probabilities, top_k)
        result = {}
        for i in range(top_k):
            result[self.categories[top_indices[i]]] = top_probabilities[i].item()
        return result
