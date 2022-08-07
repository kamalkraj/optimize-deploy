import torch
from torchvision import transforms


class Pipeline:
    def __init__(self, path) -> None:
        super().__init__()
        self.transformer = transforms.Compose(
            [
                transforms.Resize(size=(32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )
        self.label2category = {i: c for i, c in enumerate(self.classes)}

    def preprocess(self, image: torch.Tensor, device: str) -> torch.Tensor:
        return self.transformer(image).to(device)

    def postprocess(self, output: torch.Tensor, top_k: int = 10) -> dict:
        probabilities = torch.nn.functional.softmax(output, dim=0)
        top_probabilities, top_indices = torch.topk(probabilities, top_k)
        result = {}
        for i in range(top_k):
            result[self.classes[top_indices[i]]] = top_probabilities[i].item()
        return result
