import argparse
import sys

import numpy as np
import tritonclient.grpc


def load_image(img_path: str):
    """
    Loads an encoded image as an array of bytes.

    """
    return np.fromfile(img_path, dtype="uint8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        default="ensemble_python_resnet50",
        help="Model name",
    )
    parser.add_argument("--image", type=str, required=True, help="Path to the image")
    parser.add_argument(
        "--url",
        type=str,
        required=False,
        default="0.0.0.0:8001",
        help="Inference server URL. Default is localhost:8001.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output",
    )
    args = parser.parse_args()

    try:
        triton_client = tritonclient.grpc.InferenceServerClient(
            url=args.url, verbose=args.verbose
        )
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    # with open(args.label_file) as f:
    #     labels_dict = {idx: line.strip() for idx, line in enumerate(f)}

    inputs = []
    outputs = []
    input_name = "INPUT"
    output_name = "OUTPUT"
    image_data = load_image(args.image)

    inputs.append(tritonclient.grpc.InferInput(input_name, image_data.shape, "UINT8"))
    outputs.append(tritonclient.grpc.InferRequestedOutput(output_name))

    inputs[0].set_data_from_numpy(image_data)
    results = triton_client.infer(
        model_name=args.model_name, inputs=inputs, outputs=outputs
    )

    print(results.as_numpy("OUTPUT").item().decode())
