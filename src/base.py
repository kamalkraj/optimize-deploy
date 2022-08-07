from abc import ABC, abstractmethod
from typing import Union


class BaseModel(ABC):
    def __init__(self):
        super().__init__()
        self.name = "model_template"
        self.optimization_time = "constrained"  # constrained or unconstrained
        self.batch_size = 1  # batch size for sample input
        self.auxiliary_data = False  # whether to download auxiliary data

    @abstractmethod
    def get_model(self):
        raise NotImplementedError("get_model() is not implemented")

    @abstractmethod
    def get_sample_input(self, batch_size: int = 1):
        """
        input_data for the model optimization
        """
        raise NotImplementedError("get_sample_input() is not implemented")

    def get_dynamic_info(self) -> Union[dict, None]:
        """
        Dictionary containing dynamic axis information.
        """
        return None

    def get_metric_for_optimization(self) -> dict:
        """
        Dictionary containing the metric to optimize and the threshold to drop.
        """
        return {"metric": "numeric_precision", "metric_drop_ths": 0.0}

    def get_optimizer_args(self):
        """
        Dictionary containing the arguments for the optimizer.
        """
        args = {}
        args["model"] = self.get_model()
        args["input_data"] = self.get_sample_input()
        args["metric_drop_ths"] = self.get_metric_for_optimization()["metric_drop_ths"]
        args["metric"] = self.get_metric_for_optimization()["metric"]
        args["optimization_time"] = self.optimization_time
        args["dynamic_info"] = self.get_dynamic_info()
        return args

    def save_auxiliary_data(self, path: str):
        """
        Save auxiliary data to the path.
        files required for pre and post processing.
        for example, labels.txt for the imagenet dataset.
        tokenizer files for the bert model.
        """
        raise NotImplementedError("save_auxiliary_data() is not implemented")
