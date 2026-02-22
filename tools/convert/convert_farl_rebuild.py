
import argparse
import os
import sys
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from farl_backbone import FaRLBackbone, load_weights

FARL_URLS = {
    "celebm": "https://github.com/FacePerceiver/facer/releases/download/models-v1/face_parsing.farl.celebm.main_ema_181500_jit.pt",
    "lapa": "https://github.com/FacePerceiver/facer/releases/download/models-v1/face_parsing.farl.lapa.main_ema_136500_jit191.pt",
}


def download_if_missing(url: str, target_path: str) -> str:
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if not os.path.isfile(target_path):
        torch.hub.download_url_to_file(url, target_path)
    return target_path

class DynamicAvgPool(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        if isinstance(self.output_size, int):
            oh = ow = self.output_size
        else:
            oh, ow = self.output_size

        def get_k_s(size, out):
            stride = size // out
            kernel = size - (out - 1) * stride
            return kernel, stride

        kh, sh = get_k_s(h, oh)
        kw, sw = get_k_s(w, ow)
        
        return F.avg_pool2d(x, kernel_size=(kh, kw), stride=(sh, sw))

class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activate = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activate(x)
        return x

class UPerHead(nn.Module):
    def __init__(self, in_channels_list=[768, 768, 768, 768], channels=768, num_classes=19, dropout_ratio=0.1):
        super().__init__()
        self.psp_modules = nn.ModuleList()
        for i in range(4):
            pool_size = (1 if i==0 else (2 if i==1 else (3 if i==2 else 6)))
            self.psp_modules.append(nn.Sequential(
                DynamicAvgPool(pool_size),
                ConvModule(in_channels_list[-1], channels, 1)
            ))
        
        self.bottleneck = ConvModule(in_channels_list[-1] + len(self.psp_modules) * channels, channels, 3, padding=1)
        
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(len(in_channels_list) - 1): 
            self.lateral_convs.append(ConvModule(in_channels_list[i], channels, 1))
            self.fpn_convs.append(ConvModule(channels, channels, 3, padding=1))
            
        self.fpn_bottleneck = ConvModule(len(in_channels_list) * channels, channels, 3, padding=1)
        
        self.dropout = nn.Dropout2d(dropout_ratio)
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        self.align_corners = False

    def psp_forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        x = inputs[-1]
        psp_outs = [x]
        for psp in self.psp_modules:
            psp_out = psp(x)
            size = [int(x.size(2)), int(x.size(3))]
            upsampled = torch.ops.aten.upsample_bilinear2d(psp_out, size, self.align_corners, None)
            psp_outs.append(upsampled)
        
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)
        return output

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            laterals.append(lateral_conv(inputs[i]))
        
        laterals.append(self.psp_forward(inputs))
        
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = [int(laterals[i-1].size(2)), int(laterals[i-1].size(3))]
            upsampled = torch.ops.aten.upsample_bilinear2d(laterals[i], prev_shape, self.align_corners, None)
            laterals[i-1] = laterals[i-1] + upsampled
            
        fpn_outs = []
        for i, fpn_conv in enumerate(self.fpn_convs):
            fpn_outs.append(fpn_conv(laterals[i]))
        fpn_outs.append(laterals[-1])
        
        output_size = [int(fpn_outs[0].size(2)), int(fpn_outs[0].size(3))]
        for i in range(1, len(fpn_outs)):
            fpn_outs[i] = torch.ops.aten.upsample_bilinear2d(fpn_outs[i], output_size, self.align_corners, None)
            
        fpn_outs_cat = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs_cat)
        output = self.dropout(output)
        output = self.conv_seg(output)
        return output

class Rebuild(nn.Module):
    def __init__(self, backbone, head_module, out_size: List[int], num_classes=19):
        super().__init__()
        self.backbone = backbone
        self.head = UPerHead(num_classes=num_classes) 
        self.out_size = out_size
        self._copy_weights(head_module, self.head)

    def _copy_weights(self, src, dst):
        src_state = src.state_dict()
        dst_state = dst.state_dict()
        for name, param in dst_state.items():
            if name in src_state:
                if param.shape == src_state[name].shape:
                    param.copy_(src_state[name])
                else:
                    print(f"Shape mismatch for {name}: {param.shape} vs {src_state[name].shape}")
            else:
                print(f"Missing param in src: {name}")

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(images)
        # features is Tuple of 4 tensors
        # UPerHead expects List or Tuple
        logits = self.head(features)
        logits = torch.ops.aten.upsample_bilinear2d(logits, self.out_size, False, None)
        aux = torch.zeros(1, device=logits.device, dtype=logits.dtype)
        return logits, aux

def compare_outputs(model_a, model_b, dummy_input, rtol=1e-3, atol=1e-3):
    print("\n--- Comparing Outputs ---")
    with torch.no_grad():
        out_a = model_a(dummy_input)
        out_b = model_b(dummy_input)
    
    # JIT model output might be tuple or tensor, check structure
    if isinstance(out_a, tuple): out_a = out_a[0]
    if isinstance(out_b, tuple): out_b = out_b[0]
    
    diff = (out_a - out_b).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"Max Diff: {max_diff:.6f}")
    print(f"Mean Diff: {mean_diff:.6f}")
    
    # Check argmax consistency
    pred_a = out_a.argmax(dim=1)
    pred_b = out_b.argmax(dim=1)
    mismatch = (pred_a != pred_b).float().mean().item()
    print(f"Argmax Mismatch Rate: {mismatch:.2%}")
    
    if mismatch < 0.01: # Allow < 1% mismatch
        print("PASS: Outputs are consistent enough!")
        return True
    else:
        print("WARNING: Outputs differ significantly.")
        return False

def convert_checkpoint(jit_path, output_onnx_path):
    print(f"\nProcessing {jit_path}...")
    
    if not os.path.exists(jit_path):
        print(f"Error: Checkpoint not found at {jit_path}")
        return

    print("Loading JIT model to inspect parameters...")
    jit_model = torch.jit.load(jit_path, map_location="cpu")
    jit_model.eval()
    
    # Determine num_classes from JIT model structure
    try:
        num_classes = jit_model.head.head.conv_seg.weight.shape[0]
        print(f"Detected num_classes: {num_classes}")
    except:
        print("Could not detect num_classes, defaulting to 19")
        num_classes = 19
        
    print("Creating Python Backbone...")
    backbone = FaRLBackbone()
    load_weights(backbone, jit_path)
    backbone.eval()
    
    dummy = torch.zeros(1, 3, 448, 448)
    
    print("Initializing Rebuild model...")
    # Use the Python backbone
    rebuild = Rebuild(backbone, jit_model.head.head, jit_model.out_size, num_classes=num_classes)
    rebuild.eval()
    
    # Verification
    compare_outputs(jit_model, rebuild, dummy)
    
    print("Tracing Rebuild model...")
    traced_rebuild = torch.jit.trace(rebuild, dummy)
    
    print(f"Exporting to {output_onnx_path}...")
    torch.onnx.export(
        traced_rebuild,
        dummy,
        output_onnx_path,
        opset_version=12,
        input_names=["images"],
        output_names=["logits", "aux"],
        do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
        keep_initializers_as_inputs=True
    )
    print(f"Export successful: {output_onnx_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=os.path.join(REPO_ROOT, "data", "torch", "hub", "checkpoints"))
    parser.add_argument("--output_dir", default=os.path.join(REPO_ROOT, "data", "onnx"))
    parser.add_argument("--skip_farl_celebm", action="store_true")
    parser.add_argument("--skip_farl_lapa", action="store_true")
    parser.add_argument("--farl_celebm_path", default=None)
    parser.add_argument("--farl_lapa_path", default=None)
    args = parser.parse_args()

    tasks = []
    if not args.skip_farl_celebm:
        celeb_path = args.farl_celebm_path or os.path.join(args.model_dir, "face_parsing.farl.celebm.main_ema_181500_jit.pt")
        if args.farl_celebm_path is None:
            celeb_path = download_if_missing(FARL_URLS["celebm"], celeb_path)
        tasks.append((celeb_path, os.path.join(args.output_dir, "farl_celebm_448.onnx")))
    if not args.skip_farl_lapa:
        lapa_path = args.farl_lapa_path or os.path.join(args.model_dir, "face_parsing.farl.lapa.main_ema_136500_jit191.pt")
        if args.farl_lapa_path is None:
            lapa_path = download_if_missing(FARL_URLS["lapa"], lapa_path)
        tasks.append((lapa_path, os.path.join(args.output_dir, "farl_lapa_448.onnx")))

    for jit_path, onnx_path in tasks:
        convert_checkpoint(jit_path, onnx_path)

if __name__ == "__main__":
    main()
