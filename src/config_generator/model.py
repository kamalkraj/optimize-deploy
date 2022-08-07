import tritonclient.grpc.model_config_pb2 as model_config
from google.protobuf import text_format


def get_model_inputs(config):
    """
    Get the model inputs with dimensions and data type.
    """
    result = []
    batch_size = config["network_parameters"]["batch_size"]
    dynamic_info = config["network_parameters"]["dynamic_info"]
    for index, (input_name, input_info) in enumerate(
        zip(config["input_names"], config["network_parameters"]["input_infos"])
    ):
        input_size = [batch_size] + input_info["size"]
        if input_info["dtype"] == "int":
            input_data_type = model_config.DataType.TYPE_INT32
        elif input_info["dtype"] == "float":
            input_data_type = model_config.DataType.TYPE_FP32
        else:
            raise Exception(f"unknown data type: {input_info['dtype']}")
        if dynamic_info is not None:
            dynamic_info_index = dynamic_info["inputs"][index]
            for keys in dynamic_info_index:
                input_size[int(keys)] = -1
        result.append(
            model_config.ModelInput(
                name=input_name, data_type=input_data_type, dims=input_size
            )
        )
    return result


def get_model_outputs(config):
    """
    Get the model outputs with dimensions and data type.
    """
    result = []
    batch_size = config["network_parameters"]["batch_size"]
    dynamic_info = config["network_parameters"]["dynamic_info"]
    for index, (output_name, output_size) in enumerate(
        zip(
            config["output_names"],
            config["network_parameters"]["output_sizes"],
        )
    ):
        output_size = [batch_size] + output_size
        output_data_type = model_config.DataType.TYPE_FP32
        if dynamic_info is not None:
            dynamic_info_index = dynamic_info["outputs"][index]
            for keys in dynamic_info_index:
                output_size[int(keys)] = -1
        result.append(
            model_config.ModelOutput(
                name=output_name, data_type=output_data_type, dims=output_size
            )
        )
    return result


def get_model_config(config, model_name, nb_instance=1, device_kind="cuda"):
    """
    Generate the model config.
    """
    engine_type = None
    if config["module_name"].endswith("tensor_rt"):
        engine_type = "tensorrt_plan"
    elif config["module_name"].endswith("onnx"):
        engine_type = "onnxruntime_onnx"
    else:
        raise Exception(f"unknown model type: {config['module_name']}")
    if device_kind == "cuda":
        instance_kind = model_config.ModelInstanceGroup.Kind.KIND_GPU
    elif device_kind == "cpu":
        instance_kind = model_config.ModelInstanceGroup.Kind.KIND_CPU
    else:
        raise Exception(f"unknown device_kind : {device_kind}")

    model_type = engine_type.split("_")[0]
    config = model_config.ModelConfig(
        name=f"{model_name}_{model_type}",
        max_batch_size=0,
        platform=engine_type,
        default_model_filename="model.bin",
        input=get_model_inputs(config),
        output=get_model_outputs(config),
        instance_group=[
            model_config.ModelInstanceGroup(count=nb_instance, kind=instance_kind)
        ],
    )
    return (
        config.name,
        text_format.MessageToString(config),
    )
