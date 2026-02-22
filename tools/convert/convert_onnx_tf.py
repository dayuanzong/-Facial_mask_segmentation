import argparse
import os
import onnx
from onnx_tf.backend import prepare


def convert(onnx_path: str, output_dir: str):
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(onnx_path)
    model = onnx.load(onnx_path)
    tf_rep = prepare(model)
    os.makedirs(output_dir, exist_ok=True)
    tf_rep.export_graph(output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    convert(args.onnx, args.output)


if __name__ == "__main__":
    main()
