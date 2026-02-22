import os
# Force protobuf implementation to python to avoid conflicts
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Dict, Any
from itertools import product
from math import ceil
import urllib.request

import tensorflow as tf
import onnxruntime as ort

TF_DEVICE = "/CPU:0"
FARL_CELEBM_ONNX_URL = "https://huggingface.co/Kamiaajik/FaRL/blob/main/face_parsing.farl.celebm.main_ema_181500_jit.onnx"

# --- Global Initialization ---

def configure_gpu():
    global TF_DEVICE
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("TensorFlow 未检测到 GPU，已禁止 CPU 回退")
    try:
        tf.config.set_visible_devices(gpus, "GPU")
        tf.config.set_visible_devices([], "CPU")
        tf.config.set_soft_device_placement(False)
    except Exception as e:
        raise RuntimeError(f"设置可见设备失败: {e}")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            raise RuntimeError(f"设置 GPU 内存增长失败: {e}")
    TF_DEVICE = "/GPU:0"
    print(f"TensorFlow 使用设备: {TF_DEVICE}, GPU 数量: {len(gpus)}")

# Call configuration once at module level
configure_gpu()

# --- RetinaFace Post-Processing (NumPy Implementation) ---

def decode(loc, priors, variances):
    # loc: [N, 4], priors: [N, 4]
    boxes = np.concatenate(
        (
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * np.exp(loc[:, 2:] * variances[1]),
        ),
        axis=1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def decode_landm(pre, priors, variances):
    # pre: [N, 10], priors: [N, 4]
    landms = np.concatenate(
        (
            priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
        ),
        axis=1,
    )
    return landms

def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

class PriorBox:
    def __init__(self, cfg, image_size):
        self.min_sizes = cfg["min_sizes"]
        self.steps = cfg["steps"]
        self.clip = cfg["clip"]
        self.image_size = image_size
        self.feature_maps = [
            [ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)]
            for step in self.steps
        ]

    def generate_anchors(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]
        output = np.array(anchors).reshape(-1, 4)
        if self.clip:
            output = np.clip(output, 0, 1)
        return output

cfg_mnet = {
    "name": "mobilenet0.25",
    "min_sizes": [[16, 32], [64, 128], [256, 512]],
    "steps": [8, 16, 32],
    "variance": [0.1, 0.2],
    "clip": False,
    "loc_weight": 2.0,
    "gpu_train": True,
    "batch_size": 32,
    "ngpu": 1,
    "epoch": 250,
    "decay1": 190,
    "decay2": 220,
    "image_size": 640,
    "pretrain": True,
    "return_layers": {"stage1": 1, "stage2": 2, "stage3": 3},
    "in_channel": 32,
    "out_channel": 64,
}

# --- Alignment Helpers ---

def _numpy_similarity_transform_matrix(from_pts: np.ndarray, to_pts: np.ndarray) -> np.ndarray:
    mfrom = from_pts.mean(axis=0, keepdims=True)
    mto = to_pts.mean(axis=0, keepdims=True)
    a1 = ((from_pts - mfrom) ** 2).sum()
    c1 = ((to_pts - mto) * (from_pts - mfrom)).sum()
    to_delta = to_pts - mto
    from_delta = from_pts - mfrom
    c2 = (to_delta[:, 0] * from_delta[:, 1] - to_delta[:, 1] * from_delta[:, 0]).sum()
    a = c1 / a1
    b = c2 / a1
    dx = mto[0, 0] - a * mfrom[0, 0] - b * mfrom[0, 1]
    dy = mto[0, 1] + b * mfrom[0, 0] - a * mfrom[0, 1]
    matrix = np.array(
        [
            [a, b, dx],
            [-b, a, dy],
        ],
        dtype=np.float32,
    )
    return matrix

def _get_quad_np(lm: np.ndarray) -> np.ndarray:
    eye_avg = (lm[0] + lm[1]) * 0.5 + 0.5
    mouth_avg = (lm[3] + lm[4]) * 0.5 + 0.5
    eye_to_eye = lm[1] - lm[0]
    eye_to_mouth = mouth_avg - eye_avg
    x = eye_to_eye - np.array([-eye_to_mouth[1], eye_to_mouth[0]])
    x /= np.hypot(x[0], x[1])
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.array([-x[1], x[0]])
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    quad_for_coeffs = quad[[0, 3, 2, 1]]
    return quad_for_coeffs.astype(np.float32)

def _get_face_align_matrix_celebm_np(face_pts: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    quad = _get_quad_np(face_pts)
    target_size = target_shape[0]
    target_pts = np.array(
        [[0, 0], [target_size, 0], [target_size, target_size], [0, target_size]],
        dtype=np.float32,
    )
    return _numpy_similarity_transform_matrix(quad, target_pts)

# --- TF Helpers ---

def _get_tf_signature(model):
    signatures = model.signatures
    if signatures:
        return list(signatures.values())[0]
    if hasattr(model, "__call__"):
        return model
    raise RuntimeError("TensorFlow model has no callable signature")

def _run_tf_model(model, input_tensor):
    signature = _get_tf_signature(model)
    if hasattr(signature, "structured_input_signature"):
        _, inputs = signature.structured_input_signature
        if not inputs:
            return signature(input_tensor)
        key = list(inputs.keys())[0]
        return signature(**{key: input_tensor})
    return signature(input_tensor)


def _download_if_missing(url: str, target_path: str):
    if os.path.isfile(target_path):
        return
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    urllib.request.urlretrieve(url, target_path)

def _assign_retinaface_outputs(outputs):
    if isinstance(outputs, dict) and {"loc", "conf", "landms"}.issubset(outputs.keys()):
        return outputs["loc"], outputs["conf"], outputs["landms"]
    if isinstance(outputs, dict):
        values = list(outputs.values())
    else:
        values = list(outputs)
    if len(values) < 3:
        raise RuntimeError("RetinaFace TensorFlow 输出数量不足")
    def score_shape(t):
        shape = tuple(t.shape)
        return shape[-1] if shape else 0
    values_sorted = sorted(values, key=score_shape)
    conf = next(v for v in values_sorted if v.shape[-1] == 2)
    loc = next(v for v in values_sorted if v.shape[-1] == 4)
    landms = next(v for v in values_sorted if v.shape[-1] == 10)
    return loc, conf, landms

def _iter_tensors(obj):
    if isinstance(obj, tf.Tensor):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_tensors(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_tensors(v)

def _assert_gpu_tensors(obj, name):
    for t in _iter_tensors(obj):
        if "GPU" not in (t.device or ""):
            raise RuntimeError(f"{name} 未在 GPU 上执行: {t.device}")

# --- Main Inference ---

def inference(input_path, output_dir, detector_dir, parser_dir):
    image = np.array(Image.open(input_path).convert("RGB"))
    
    # Check if models are ONNX to avoid unnecessary TF tensor creation
    detector_is_onnx = detector_dir.lower().endswith(".onnx") and os.path.isfile(detector_dir)
    parser_is_onnx = parser_dir.lower().endswith(".onnx")
    if not parser_is_onnx:
        saved_pb = os.path.join(parser_dir, "saved_model.pb")
        saved_pbtxt = os.path.join(parser_dir, "saved_model.pbtxt")
        if not (os.path.exists(saved_pb) or os.path.exists(saved_pbtxt)):
            parser_is_onnx = True

    # Lazy tensor creation only if needed by TF models
    image_tensor = None
    if not detector_is_onnx or not parser_is_onnx:
        with tf.device(TF_DEVICE):
            image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        _assert_gpu_tensors(image_tensor, "image_tensor")

    detector_input_name = None
    input_is_nhwc = False
    if detector_is_onnx:
        ort.set_default_logger_severity(3)
        providers = ["CPUExecutionProvider"]
        if os.environ.get("ORT_CUDA", "0") == "1":
             providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        detector = ort.InferenceSession(detector_dir, providers=providers)
        detector_input_name = detector.get_inputs()[0].name
    else:
        detector = tf.saved_model.load(detector_dir)
        try:
            sig = _get_tf_signature(detector)
            if hasattr(sig, "structured_input_signature"):
                input_spec = sig.structured_input_signature[1]
                if input_spec:
                    key = list(input_spec.keys())[0]
                    shape = input_spec[key].shape
                    if shape[-1] == 3:
                        input_is_nhwc = True
                    elif shape[1] == 3:
                        input_is_nhwc = False
        except Exception as e:
            print(f"Failed to inspect detector signature: {e}, assuming NCHW.")

    parser_is_onnx = parser_dir.lower().endswith(".onnx")
    if not parser_is_onnx:
        saved_pb = os.path.join(parser_dir, "saved_model.pb")
        saved_pbtxt = os.path.join(parser_dir, "saved_model.pbtxt")
        if not (os.path.exists(saved_pb) or os.path.exists(saved_pbtxt)):
            parser_is_onnx = True
            
    if parser_is_onnx:
        ort.set_default_logger_severity(3)
        providers = ["CPUExecutionProvider"]
        if os.environ.get("ORT_CUDA", "0") == "1":
             providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        parser = ort.InferenceSession(parser_dir, providers=providers)
        parser_input_name = parser.get_inputs()[0].name
    else:
        parser = tf.saved_model.load(parser_dir)
    
    if detector_is_onnx:
        input_tensor = image.astype(np.float32).transpose(2, 0, 1)[None, ...]
        mean = np.array([104.0, 117.0, 123.0], dtype=np.float32).reshape(1, 3, 1, 1)
        input_tensor = input_tensor - mean
        outputs = detector.run(None, {detector_input_name: input_tensor})
        loc, conf, landms = _assign_retinaface_outputs(outputs)
    else:
        with tf.device(TF_DEVICE):
            if input_is_nhwc:
                input_tensor = tf.expand_dims(image_tensor, axis=0)
            else:
                input_tensor = tf.expand_dims(tf.transpose(image_tensor, [2, 0, 1]), axis=0)

            mean = tf.constant([104.0, 117.0, 123.0], dtype=tf.float32)
            if input_is_nhwc:
                input_tensor = input_tensor - mean[None, None, None, :]
            else:
                input_tensor = input_tensor - mean[None, :, None, None]
            outputs = _run_tf_model(detector, input_tensor)
        _assert_gpu_tensors(input_tensor, "detector_input")
        _assert_gpu_tensors(outputs, "detector_output")
        loc, conf, landms = _assign_retinaface_outputs(outputs)
        loc = loc.numpy()
        conf = conf.numpy()
        landms = landms.numpy()
    
    loc_abs = np.abs(loc).mean() > 10.0
    if loc_abs:
        print("WARNING: Detector outputs seem to be absolute coordinates! Skipping decoding.")
        boxes = loc[0]
        scores = conf[0][:, 1] if conf.ndim == 3 else conf[:, 1]
        landms_dec = landms[0]
    else:
        if detector_is_onnx:
            h, w = image.shape[:2]
        elif input_is_nhwc:
            h = int(input_tensor.shape[1])
            w = int(input_tensor.shape[2])
        else:
            h = int(input_tensor.shape[2])
            w = int(input_tensor.shape[3])
        scale = np.array([w, h, w, h], dtype=np.float32)
        scale1 = np.array([w, h, w, h, w, h, w, h, w, h], dtype=np.float32)
        priorbox = PriorBox(cfg_mnet, image_size=(int(h), int(w)))
        prior_data = priorbox.generate_anchors()
        boxes = decode(loc[0], prior_data, cfg_mnet["variance"])
        boxes = boxes * scale
        scores = conf[0][:, 1]
        landms_dec = decode_landm(landms[0], prior_data, cfg_mnet["variance"])
        landms_dec = landms_dec * scale1
    inds = np.where(scores > 0.8)[0]
    if inds.size == 0:
        return
    boxes = boxes[inds]
    landms_dec = landms_dec[inds]
    scores = scores[inds]
    order = scores.argsort()[::-1][:5000]
    boxes = boxes[order]
    landms_dec = landms_dec[order]
    scores = scores[order]
    dets = np.hstack((boxes, scores[:, None])).astype(np.float32)
    keep = nms(dets, 0.4)
    keep = np.array(keep, dtype=np.int64)
    dets = dets[keep][:750]
    landms_dec = landms_dec[keep][:750]
    if dets.shape[0] == 0:
        return
        
    label_names = [
        "background", "neck", "face", "cloth", "rr", "lr", "rb", "lb", "re",
        "le", "nose", "imouth", "llip", "ulip", "hair", "eyeg", "hat", "earr", "neck_l",
    ]
    allowed_names = ["face", "rb", "lb", "re", "le", "nose", "imouth", "llip", "ulip"]
    allowed_idx = [i for i, n in enumerate(label_names) if n in allowed_names]
    
    masks = []
    for lm in landms_dec:
        lm = lm.reshape(5, 2)
        matrix = _get_face_align_matrix_celebm_np(lm, (448, 448))
        inv = cv2.invertAffineTransform(matrix)
        # Use WARP_INVERSE_MAP for clarity: inv is Dst(448)->Src(1024) mapping
        warped = cv2.warpAffine(image, inv, (448, 448), flags=cv2.WARP_INVERSE_MAP)
        
        if parser_is_onnx:
            # ONNX expects NCHW, normalized 0-1
            warped_tensor = (warped.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
            seg_logits = parser.run(None, {parser_input_name: warped_tensor})[0]
        else:
            # TF SavedModel expects NHWC (usually), normalized 0-1
            with tf.device(TF_DEVICE):
                warped_tensor = tf.convert_to_tensor(warped, dtype=tf.float32) / 255.0
                warped_tensor = tf.expand_dims(warped_tensor, axis=0)
                parser_out = _run_tf_model(parser, warped_tensor)
            _assert_gpu_tensors(warped_tensor, "parser_input")
            _assert_gpu_tensors(parser_out, "parser_output")
            if isinstance(parser_out, dict):
                seg_logits = list(parser_out.values())[0].numpy()
            else:
                seg_logits = parser_out.numpy()
            
            if seg_logits.ndim == 4 and seg_logits.shape[-1] == len(label_names):
                 seg_logits = np.transpose(seg_logits, (0, 3, 1, 2))

        if seg_logits.ndim == 4 and seg_logits.shape[1] == len(label_names):
            seg_logits = np.transpose(seg_logits, (0, 2, 3, 1))
        
        # Resize logits if shape mismatch (e.g. model outputs 512x512 for 448x448 input)
        # NOTE: FaRL ONNX model (and potentially others) may output feature maps with a different resolution 
        # (e.g., 512x512) than the input image (e.g., 448x448) due to padding or architectural differences 
        # (e.g. ViT patch size alignment). The alignment matrix is calculated based on the input resolution. 
        # Therefore, we must resize the output logits to match the input resolution before unwarping to ensure 
        # correct spatial mapping.
        if seg_logits.shape[1] != warped.shape[0] or seg_logits.shape[2] != warped.shape[1]:
            seg_logits_resized = []
            for i in range(seg_logits.shape[-1]):
                ch = seg_logits[0, :, :, i]
                ch_resized = cv2.resize(ch, (warped.shape[1], warped.shape[0]), interpolation=cv2.INTER_LINEAR)
                seg_logits_resized.append(ch_resized)
            seg_logits = np.stack(seg_logits_resized, axis=-1)[None, ...]
        
        # Unwarp logits to original image size using matrix (Dst 1024 -> Src 448 mapping)
        logits_full = []
        for i in range(seg_logits.shape[-1]):
            ch = seg_logits[0, :, :, i]
            # Use WARP_INVERSE_MAP: matrix is Dst(1024)->Src(448) mapping
            ch_full = cv2.warpAffine(ch, matrix, (image.shape[1], image.shape[0]), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
            logits_full.append(ch_full)
        logits_full = np.stack(logits_full, axis=-1)
        
        seg_mask = np.argmax(logits_full, axis=-1).astype(np.uint8)
        mask = np.isin(seg_mask, allowed_idx).astype(np.uint8) * 255
        masks.append(mask)
        
    if not masks:
        return
    mask = np.maximum.reduce(masks)
    os.makedirs(output_dir, exist_ok=True)
    overlay = image.copy()
    color = np.array([0, 255, 0], dtype=np.uint8)
    alpha = 0.4
    overlay[mask > 0] = (overlay[mask > 0] * (1 - alpha) + color * alpha).astype(np.uint8)
    output_name = os.path.splitext(os.path.basename(input_path))[0] + "_tf.png"
    out_overlay = os.path.join(output_dir, output_name)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_overlay, overlay_bgr)

def inference_dir(input_dir, output_dir, detector_dir, parser_dir):
    if os.path.isfile(input_dir):
        inference(input_dir, output_dir, detector_dir, parser_dir)
        return
    for name in os.listdir(input_dir):
        if not name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            continue
        if "_mask_" in name.lower():
            continue
        input_path = os.path.join(input_dir, name)
        try:
            inference(input_path, output_dir, detector_dir, parser_dir)
        except Exception as e:
            print(f"Error processing {input_path}: {e}")

if __name__ == "__main__":
    input_dir = r"D:\AI\temp\_internal\ExtractTool\temp\3DDFA-V3\data\input"
    output_dir = r"D:\AI\temp\_internal\ExtractTool\temp\3DDFA-V3\data\results"
    detector_dir = r"D:\AI\temp\_internal\ExtractTool\temp\3DDFA-V3\data\onnx\retinaface_mobilenet.onnx"
    parser_dir = r"D:\AI\temp\_internal\ExtractTool\temp\3DDFA-V3\data\onnx\farl_celebm_448.onnx"

    _download_if_missing(FARL_CELEBM_ONNX_URL, parser_dir)
    inference_dir(
        input_dir,
        output_dir,
        detector_dir=detector_dir,
        parser_dir=parser_dir,
    )
