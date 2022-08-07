import argparse
import json
import logging
import os
import shutil
import sys

from nebullvm.api.functions import optimize_model

from config_generator.model import get_model_config
from config_generator.pipeline import generate_pipeline_config
from config_generator.utils import get_input_names, get_output_names

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimize and generate a model for Triton serving"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model to optimize",
    )
    parser.add_argument(
        "--optimized_model_path",
        type=str,
        required=True,
        help="Path to the optimized model",
    )
    parser.add_argument(
        "--triton_model_path",
        type=str,
        required=True,
        help="Path to the Triton model",
    )
    parser.add_argument(
        "--task",
        choices=["image_classification", "text_classification"],
        required=True,
        help="Task to optimize for",
    )
    parser.add_argument(
        "--ignore_compilers",
        choices=["tensor RT", "onnxruntime"],
        default=None,
        help="Ignore compilers",
    )
    return parser.parse_args()


def get_model(model_path: str):
    # import the model from the model path
    sys.path.append(os.path.dirname(model_path))
    from model import Model

    # create the model
    model = Model()

    return model


def main():
    args = parse_args()
    model_path = args.model_path
    output_path = args.optimized_model_path
    triton_model_path = args.triton_model_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model = get_model(model_path)

    optimize_model_args = model.get_optimizer_args()
    if args.ignore_compilers is not None:
        optimize_model_args["ignore_compilers"] = args.ignore_compilers.split(",")

    # optimize the model
    optimized_model = optimize_model(**optimize_model_args)

    # save the optimized model to the output path
    output_path = os.path.join(output_path, model.name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    optimized_model.save(output_path)

    logger.info(f"Optimized model saved to {output_path}")

    # save the optimized model to the Triton model path
    config_path = os.path.join(output_path, "optimized_model", "metadata.json")
    nebullvm_config = json.load(open(config_path))
    # generate and save model config for Triton
    triton_folder_name, triton_config = get_model_config(nebullvm_config, model.name)
    triton_model_path = os.path.join(triton_model_path, triton_folder_name)
    if os.path.exists(triton_model_path):
        os.remove(triton_model_path)
    os.makedirs(triton_model_path)
    triton_config_path = os.path.join(triton_model_path, "config.pbtxt")
    with open(triton_config_path, "w") as f:
        f.write(triton_config)
    # rename and move the optimized model to the Triton model path
    os.mkdir(os.path.join(triton_model_path, "1"))
    files = os.listdir(os.path.join(output_path, "optimized_model"))
    files.remove("metadata.json")
    shutil.copy(
        os.path.join(output_path, "optimized_model", files[0]),
        os.path.join(triton_model_path, "1", "model.bin"),
    )
    logger.info(f"Triton model saved to {triton_model_path}")

    # generate the pipeline config
    template_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "templates", args.task
    )
    pipeline_folder_name = f"{model.name}_pipeline"
    pipeline_path = os.path.join(args.triton_model_path, pipeline_folder_name)
    if os.path.exists(pipeline_path):
        shutil.rmtree(pipeline_path)
    os.makedirs(pipeline_path)
    if args.task == "image_classification":
        pbtxt_args = {"name": pipeline_folder_name}
        model_args = {
            "model_name": triton_folder_name,
            "input_name": get_input_names(nebullvm_config)[0],
            "output_name": get_output_names(nebullvm_config)[0],
        }

    elif args.task == "text_classification":
        raise NotImplementedError("Text classification not implemented")
    else:
        raise ValueError(f"Unknown task {args.task}")

    pbtxt_generated, model_generated = generate_pipeline_config(
        template_path, pbtxt_args, model_args
    )

    pbtxt_path = os.path.join(pipeline_path, "config.pbtxt")
    with open(pbtxt_path, "w") as f:
        f.write(pbtxt_generated)
    model_path = os.path.join(pipeline_path, "1")
    os.makedirs(model_path)
    model_file_path = os.path.join(model_path, "model.py")
    with open(model_file_path, "w") as f:
        f.write(model_generated)

    # move the pipeline code to the Triton model path
    pipeline_source_path = os.path.join(args.model_path, "pipeline.py")
    shutil.copy(pipeline_source_path, model_path)

    # Download any auxiliary files required by the model pre-post processing
    if model.auxiliary_data:
        auxiliary_data_path = os.path.join(pipeline_path, "1")
        os.makedirs(auxiliary_data_path, exist_ok=True)
        model.save_auxiliary_data(auxiliary_data_path)

    logger.info(f"Pipeline config saved to {pipeline_path}")


if __name__ == "__main__":
    main()
