import argparse
import os
import sys
from typing import List, Tuple
import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

FARL_URLS = {
    "celebm": "https://github.com/FacePerceiver/facer/releases/download/models-v1/face_parsing.farl.celebm.main_ema_181500_jit.pt",
    "lapa": "https://github.com/FacePerceiver/facer/releases/download/models-v1/face_parsing.farl.lapa.main_ema_136500_jit191.pt",
}


def download_if_missing(url: str, target_path: str) -> str:
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if not os.path.isfile(target_path):
        torch.hub.download_url_to_file(url, target_path)
    return target_path

class Rebuild(nn.Module):
    def __init__(self, backbone, head, out_size: List[int]):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.out_size = out_size

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features, _ = self.backbone(images)
        logits = self.head(features)
        # Using functional interpolate directly which should be scriptable
        # Explicitly use upsample_bilinear2d to avoid aten::__interpolate
        # out_size must be List[int]
        logits = torch.ops.aten.upsample_bilinear2d(logits, self.out_size, False, None)
        aux = torch.zeros(1, device=logits.device, dtype=logits.dtype)
        return logits, aux

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["celebm", "lapa"], default="celebm")
    parser.add_argument("--jit_path", default=None)
    parser.add_argument("--model_dir", default=os.path.join(REPO_ROOT, "data", "torch", "hub", "checkpoints"))
    parser.add_argument("--output_dir", default=os.path.join(REPO_ROOT, "data", "onnx"))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    default_jit = f"face_parsing.farl.{args.model}.main_ema_181500_jit.pt" if args.model == "celebm" else "face_parsing.farl.lapa.main_ema_136500_jit191.pt"
    jit_path = args.jit_path or os.path.join(args.model_dir, default_jit)
    if args.jit_path is None:
        jit_path = download_if_missing(FARL_URLS[args.model], jit_path)
    print(f"Loading {jit_path}...")
    net = torch.jit.load(jit_path, map_location="cpu")
    net.eval()
    
    print("Rebuilding model...")
    rebuild = Rebuild(net.backbone, net.head, net.out_size)
    rebuild.eval()
    
    print("Scripting model...")
    scripted = torch.jit.script(rebuild)
    
    output_name = f"farl_{args.model}_448.onnx"
    output_path = args.output or os.path.join(args.output_dir, output_name)
    dummy = torch.zeros(1, 3, 448, 448)
    
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        scripted,
        dummy,
        output_path,
        opset_version=13,
        input_names=["images"],
        output_names=["logits", "aux"],
        do_constant_folding=False,
        training=torch.onnx.TrainingMode.EVAL,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX
    )
    print("Export successful!")

if __name__ == "__main__":
    main()
