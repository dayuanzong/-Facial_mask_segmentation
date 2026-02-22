import argparse
import importlib
import os
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

torch = None
RetinaFace = None
cfg_mnet = None
load_model = None
pretrained_urls = None

FARL_URLS = {
    "celebm": "https://github.com/FacePerceiver/facer/releases/download/models-v1/face_parsing.farl.celebm.main_ema_181500_jit.pt",
    "lapa": "https://github.com/FacePerceiver/facer/releases/download/models-v1/face_parsing.farl.lapa.main_ema_136500_jit191.pt",
}


def download_if_missing(url: str, target_path: str) -> str:
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if not os.path.isfile(target_path):
        torch.hub.download_url_to_file(url, target_path)
    return target_path


def ensure_deps():
    global torch, RetinaFace, cfg_mnet, load_model, pretrained_urls
    if torch is None:
        try:
            torch = importlib.import_module("torch")
        except Exception:
            requirements_path = os.path.join(REPO_ROOT, "requirements.txt")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
            torch = importlib.import_module("torch")
    if RetinaFace is None:
        detector = importlib.import_module("retinaface_detector")
        RetinaFace = detector.RetinaFace
        cfg_mnet = detector.cfg_mnet
        load_model = detector.load_model
        pretrained_urls = detector.pretrained_urls


def export_retinaface(weights_path: str, output_path: str, image_size: int):
    net = RetinaFace(cfg=cfg_mnet, phase="test")
    net = load_model(net, weights_path, True, network="mobilenet")
    net.eval()
    dummy = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.onnx.export(
        net,
        dummy,
        output_path,
        opset_version=11,
        input_names=["images"],
        output_names=["loc", "conf", "landms"],
        dynamic_axes={
            "images": {0: "batch", 2: "height", 3: "width"},
            "loc": {0: "batch"},
            "conf": {0: "batch"},
            "landms": {0: "batch"},
        },
    )


def export_farl(jit_path: str, output_path: str):
    net = torch.jit.load(jit_path, map_location="cpu")
    net.eval()
    dummy = torch.zeros(1, 3, 448, 448, dtype=torch.float32)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if hasattr(torch, "_C"):
        if hasattr(torch._C, "_set_graph_executor_optimize"):
            torch._C._set_graph_executor_optimize(False)
        if hasattr(torch._C, "_jit_set_profiling_mode"):
            torch._C._jit_set_profiling_mode(False)
        if hasattr(torch._C, "_jit_set_profiling_executor"):
            torch._C._jit_set_profiling_executor(False)
    class Rebuild(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.backbone = module.backbone
            self.head = module.head
            self.out_size = module.out_size

        def forward(self, images):
            features, _ = self.backbone(images)
            logits = self.head(features)
            logits = torch.nn.functional.interpolate(
                logits,
                size=self.out_size,
                mode="bilinear",
                align_corners=False,
            )
            aux = torch.zeros(1, device=logits.device, dtype=logits.dtype)
            return logits, aux

    rebuild = Rebuild(net)
    rebuild.eval()
    rebuild_traced = torch.jit.trace(rebuild, dummy, check_trace=False, strict=False)
    traced = torch.jit.trace(net, dummy, check_trace=False)
    last_exc = None
    if hasattr(torch.onnx, "dynamo_export"):
        class DynamoWrapper(torch.nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, images):
                return self.module(images)

        try:
            exported = torch.onnx.dynamo_export(DynamoWrapper(net), dummy)
            exported.save(output_path)
            return
        except Exception as exc:
            last_exc = exc
    for opset in (13, 12, 11):
        try:
            torch.onnx.export(
                rebuild_traced,
                dummy,
                output_path,
                opset_version=opset,
                input_names=["images"],
                output_names=["logits", "aux"],
                do_constant_folding=False,
                training=torch.onnx.TrainingMode.EVAL,
                operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
            )
            return
        except Exception as exc:
            last_exc = exc
    for opset in (13, 12, 11):
        try:
            torch.onnx.export(
                traced,
                dummy,
                output_path,
                opset_version=opset,
                input_names=["images"],
                output_names=["logits", "aux"],
                do_constant_folding=False,
                training=torch.onnx.TrainingMode.EVAL,
                operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
            )
            return
        except Exception as exc:
            last_exc = exc
    for opset in (13, 12, 11):
        try:
            torch.onnx.export(
                net,
                dummy,
                output_path,
                opset_version=opset,
                input_names=["images"],
                output_names=["logits", "aux"],
                do_constant_folding=False,
                training=torch.onnx.TrainingMode.EVAL,
                operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
            )
            return
        except Exception as exc:
            last_exc = exc
    class Wrapper(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, images):
            return self.module(images)

    wrapper = Wrapper(net)
    for opset in (13, 12, 11):
        try:
            torch.onnx.export(
                wrapper,
                dummy,
                output_path,
                opset_version=opset,
                input_names=["images"],
                output_names=["logits", "aux"],
                do_constant_folding=False,
                training=torch.onnx.TrainingMode.EVAL,
                operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
            )
            return
        except Exception as exc:
            last_exc = exc
    if last_exc:
        raise last_exc


def prompt_selection():
    print("请选择要转换的模型：")
    print("1. RetinaFace mobilenet")
    print("2. FaRL celebM")
    print("3. FaRL LaPa")
    print("4. 全部")
    while True:
        choice = input("输入数字: ").strip()
        if choice in {"1", "2", "3", "4"}:
            return choice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=os.path.join(REPO_ROOT, "data", "onnx"))
    parser.add_argument("--model_dir", default=os.path.join(REPO_ROOT, "data", "torch", "hub", "checkpoints"))
    parser.add_argument("--image_size", type=int, default=640)
    parser.add_argument("--with_retinaface", action="store_true")
    parser.add_argument("--skip_farl_celebm", action="store_true")
    parser.add_argument("--skip_farl_lapa", action="store_true")
    parser.add_argument("--retinaface_weights", default=None)
    parser.add_argument("--farl_celebm_path", default=None)
    parser.add_argument("--farl_lapa_path", default=None)
    args = parser.parse_args()

    output_dir = args.output_dir
    ensure_deps()

    use_prompt = not any(
        [
            args.with_retinaface,
            args.skip_farl_celebm,
            args.skip_farl_lapa,
            args.retinaface_weights,
            args.farl_celebm_path,
            args.farl_lapa_path,
        ]
    )
    if use_prompt:
        choice = prompt_selection()
        if choice == "1":
            args.with_retinaface = True
            args.skip_farl_celebm = True
            args.skip_farl_lapa = True
        elif choice == "2":
            args.skip_farl_lapa = True
        elif choice == "3":
            args.skip_farl_celebm = True
        elif choice == "4":
            args.with_retinaface = True

    if args.with_retinaface:
        retina_path = args.retinaface_weights or os.path.join(args.model_dir, "mobilenet0.25_Final.pth")
        if args.retinaface_weights is None:
            retina_path = download_if_missing(pretrained_urls["mobilenet"], retina_path)
        try:
            export_retinaface(retina_path, os.path.join(output_dir, "retinaface_mobilenet.onnx"), args.image_size)
        except Exception as exc:
            print(f"retinaface export failed: {exc}")

    if not args.skip_farl_celebm:
        farl_celebm = args.farl_celebm_path or os.path.join(args.model_dir, "face_parsing.farl.celebm.main_ema_181500_jit.pt")
        if args.farl_celebm_path is None:
            farl_celebm = download_if_missing(FARL_URLS["celebm"], farl_celebm)
        try:
            export_farl(farl_celebm, os.path.join(output_dir, "farl_celebm_448.onnx"))
        except Exception as exc:
            print(f"farl celebM export failed: {exc}")
    if not args.skip_farl_lapa:
        farl_lapa = args.farl_lapa_path or os.path.join(args.model_dir, "face_parsing.farl.lapa.main_ema_136500_jit191.pt")
        if args.farl_lapa_path is None:
            farl_lapa = download_if_missing(FARL_URLS["lapa"], farl_lapa)
        try:
            export_farl(farl_lapa, os.path.join(output_dir, "farl_lapa_448.onnx"))
        except Exception as exc:
            print(f"farl lapa export failed: {exc}")


if __name__ == "__main__":
    main()
