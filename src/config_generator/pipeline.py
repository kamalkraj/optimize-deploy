from jinja2 import Environment, FileSystemLoader


def generate_pipeline_config(template_dir, pbtxt_args: dict, model_args: dict):
    """
    Generate the pipeline config.
    """
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("config.pbtxt.jinja2")
    config = template.render(**pbtxt_args)
    template = env.get_template("model.py.jinja2")
    model = template.render(**model_args)
    return config, model
