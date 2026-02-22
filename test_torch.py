import os
import torch
import numpy as np
import cv2
from PIL import Image
from retinaface_detector import RetinaFaceDetector
from farl_face_parser import FaRLFaceParser
from typing import Optional, Tuple

FARL_CELEBM_URL = "https://github.com/FacePerceiver/facer/releases/download/models-v1/face_parsing.farl.celebm.main_ema_181500_jit.pt"


def download_if_missing(url: str, target_path: str):
    if os.path.isfile(target_path):
        return
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    torch.hub.download_url_to_file(url, target_path)

def read_hwc(path: str) -> torch.Tensor:
    image = Image.open(path)
    np_image = np.array(image.convert("RGB"))
    return torch.from_numpy(np_image)

def hwc2bchw(images: torch.Tensor) -> torch.Tensor:
    return images.unsqueeze(0).permute(0, 3, 1, 2)

def _split_name(name: str) -> Tuple[str, Optional[str]]:
    if "/" in name:
        detector_type, conf_name = name.split("/", 1)
    else:
        detector_type, conf_name = name, None
    return detector_type, conf_name

def make_face_detector(name: str, device: torch.device, **kwargs):
    detector_type, conf_name = _split_name(name)
    if detector_type == "retinaface":
        return RetinaFaceDetector(conf_name, device=device, **kwargs)
    raise RuntimeError(f"Unknown detector type: {detector_type}")

def make_face_parser(name: str, device: torch.device, **kwargs):
    parser_type, conf_name = _split_name(name)
    if parser_type == "farl":
        return FaRLFaceParser(conf_name, device=device, **kwargs).to(device)
    raise RuntimeError(f"Unknown parser type: {parser_type}")

def inference_torch(input_path, output_dir, model_path):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，已禁用 CPU 回退")
    device = torch.device("cuda")

    image = hwc2bchw(read_hwc(input_path)).to(device)

    detector = make_face_detector("retinaface/mobilenet", device=device)
    with torch.inference_mode():
        faces = detector(image)
    
    if faces["image_ids"].numel() == 0:
        return
    if faces["image_ids"].dtype != torch.long:
        faces["image_ids"] = faces["image_ids"].long()

    face_parser = make_face_parser(
        "farl/celebm/448",
        device,
        model_path=model_path,
    )

    with torch.inference_mode():
        faces = face_parser(image, faces)

    seg_logits = faces["seg"]["logits"]
    label_names = faces["seg"]["label_names"]
    seg_probs = seg_logits.softmax(dim=1)
    seg_mask = seg_probs.argmax(dim=1).cpu().numpy().astype(np.uint8)
    allowed_names = ["face", "rb", "lb", "re", "le", "nose", "imouth", "llip", "ulip"]
    allowed_idx = [i for i, n in enumerate(label_names) if n in allowed_names]
    mask = np.isin(seg_mask.max(axis=0), allowed_idx).astype(np.uint8) * 255

    img = cv2.imread(input_path)
    os.makedirs(output_dir, exist_ok=True)
    overlay = img.copy()
    color = np.array([0, 255, 0], dtype=np.uint8)
    alpha = 0.4
    overlay[mask > 0] = (overlay[mask > 0] * (1 - alpha) + color * alpha).astype(np.uint8)

    output_name = os.path.splitext(os.path.basename(input_path))[0] + "_torch.png"
    out_overlay = os.path.join(output_dir, output_name)
    cv2.imwrite(out_overlay, overlay)

def inference_dir_torch(input_dir, output_dir, model_path):
    if os.path.isfile(input_dir):
        inference_torch(input_dir, output_dir, model_path)
        return
    for name in os.listdir(input_dir):
        if not name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            continue
        if "_mask_" in name.lower():
            continue
        input_path = os.path.join(input_dir, name)
        try:
            inference_torch(input_path, output_dir, model_path)
        except Exception as e:
            print(f"Error processing {input_path}: {e}")

if __name__ == "__main__":
    os.environ["TORCH_HOME"] = r"D:\AI\temp\_internal\ExtractTool\temp\3DDFA-V3\data\torch"
    model_path = r"D:\AI\temp\_internal\ExtractTool\temp\3DDFA-V3\data\torch\hub\checkpoints\face_parsing.farl.celebm.main_ema_181500_jit.pt"
    download_if_missing(FARL_CELEBM_URL, model_path)
    inference_dir_torch(
        r"D:\AI\temp\_internal\ExtractTool\temp\3DDFA-V3\data\input",
        r"D:\AI\temp\_internal\ExtractTool\temp\3DDFA-V3\data\results",
        model_path,
    )
