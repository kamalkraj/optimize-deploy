def get_input_names(config):
    """
    Get the input names from the config.
    """
    input_names = []
    for input_name in config["input_names"]:
        input_names.append(input_name)
    return input_names


def get_output_names(config):
    """
    Get the output names from the config.
    """
    output_names = []
    for output_name in config["output_names"]:
        output_names.append(output_name)
    return output_names
