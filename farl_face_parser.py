from typing import Optional, Dict, Any, Tuple, Callable, Union, List
import functools
import os
import errno
import sys
import json
from urllib.parse import urlparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import validators


class FaceParser(nn.Module):
    pass


def is_github_url(url: str):
    return ("blob" in url or "raw" in url) and url.startswith("https://github.com/")


def get_github_components(url: str):
    organisation, repository, blob_or_raw, branch, *path = url[len("https://github.com/") :].split("/")
    assert blob_or_raw in {"blob", "raw"}
    return organisation, repository, branch, "/".join(path)


def download_from_github(to_path, organisation, repository, file_path, branch="main", username=None, access_token=None):
    if username is not None:
        assert access_token is not None
        auth = (username, access_token)
    else:
        auth = None
    import requests
    r = requests.get(
        f"https://api.github.com/repos/{organisation}/{repository}/contents/{file_path}?ref={branch}",
        auth=auth,
    )
    data = json.loads(r.content)
    torch.hub.download_url_to_file(data["download_url"], to_path)


def download_url_to_file(url, dst, **kwargs):
    if is_github_url(url):
        org, rep, branch, path = get_github_components(url)
        download_from_github(dst, org, rep, path, branch, kwargs.get("username", None), kwargs.get("access_token", None))
    else:
        torch.hub.download_url_to_file(url, dst)


def download_jit(url_or_paths: Union[str, List[str]], model_dir=None, map_location=None, jit=True, **kwargs):
    if isinstance(url_or_paths, str):
        url_or_paths = [url_or_paths]
    for url_or_path in url_or_paths:
        try:
            if validators.url(url_or_path):
                url = url_or_path
                if model_dir is None:
                    if hasattr(torch.hub, "get_dir"):
                        hub_dir = torch.hub.get_dir()
                    else:
                        hub_dir = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub")
                    model_dir = os.path.join(hub_dir, "checkpoints")
                try:
                    os.makedirs(model_dir)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
                parts = urlparse(url)
                filename = os.path.basename(parts.path)
                cached_file = os.path.join(model_dir, filename)
                if not os.path.exists(cached_file):
                    sys.stderr.write(f'Downloading: "{url}" to {cached_file}\n')
                    download_url_to_file(url, cached_file)
            else:
                cached_file = url_or_path
            if jit:
                return torch.jit.load(cached_file, map_location=map_location, **kwargs)
            return torch.load(cached_file, map_location=map_location, **kwargs)
        except:
            sys.stderr.write(f"failed downloading from {url_or_path}\n")
            raise
    raise RuntimeError("failed to download jit models from all given urls")


def get_similarity_transform_matrix(from_pts: torch.Tensor, to_pts: torch.Tensor) -> torch.Tensor:
    mfrom = from_pts.mean(dim=1, keepdim=True)
    mto = to_pts.mean(dim=1, keepdim=True)
    a1 = (from_pts - mfrom).square().sum([1, 2], keepdim=False)
    c1 = ((to_pts - mto) * (from_pts - mfrom)).sum([1, 2], keepdim=False)
    to_delta = to_pts - mto
    from_delta = from_pts - mfrom
    c2 = (to_delta[:, :, 0] * from_delta[:, :, 1] - to_delta[:, :, 1] * from_delta[:, :, 0]).sum(
        [1], keepdim=False
    )
    a = c1 / a1
    b = c2 / a1
    dx = mto[:, 0, 0] - a * mfrom[:, 0, 0] - b * mfrom[:, 0, 1]
    dy = mto[:, 0, 1] + b * mfrom[:, 0, 0] - a * mfrom[:, 0, 1]
    ones_pl = torch.ones_like(a1)
    zeros_pl = torch.zeros_like(a1)
    return torch.stack(
        [
            a,
            b,
            dx,
            -b,
            a,
            dy,
            zeros_pl,
            zeros_pl,
            ones_pl,
        ],
        dim=-1,
    ).reshape(-1, 3, 3)


@functools.lru_cache()
def _standard_face_pts():
    pts = torch.tensor(
        [196.0, 226.0, 316.0, 226.0, 256.0, 286.0, 220.0, 360.4, 292.0, 360.4],
        dtype=torch.float32,
    ) / 256.0 - 1.0
    return torch.reshape(pts, (5, 2))


def get_face_align_matrix(
    face_pts: torch.Tensor,
    target_shape: Tuple[int, int],
    target_face_scale: float = 1.0,
    offset_xy: Optional[Tuple[float, float]] = None,
    target_pts: Optional[torch.Tensor] = None,
):
    if target_pts is None:
        with torch.no_grad():
            std_pts = _standard_face_pts().to(face_pts)
            h, w, *_ = target_shape
            target_pts = (std_pts * target_face_scale + 1) * torch.tensor([w - 1, h - 1]).to(face_pts) / 2.0
            if offset_xy is not None:
                target_pts[:, 0] += offset_xy[0]
                target_pts[:, 1] += offset_xy[1]
    else:
        target_pts = target_pts.to(face_pts)
    if target_pts.dim() == 2:
        target_pts = target_pts.unsqueeze(0)
    if target_pts.size(0) == 1:
        target_pts = target_pts.broadcast_to(face_pts.shape)
    assert target_pts.shape == face_pts.shape
    return get_similarity_transform_matrix(face_pts, target_pts)


def rot90(v):
    return np.array([-v[1], v[0]])


def get_quad(lm: torch.Tensor):
    lm = lm.detach().cpu().numpy()
    eye_avg = (lm[0] + lm[1]) * 0.5 + 0.5
    mouth_avg = (lm[3] + lm[4]) * 0.5 + 0.5
    eye_to_eye = lm[1] - lm[0]
    eye_to_mouth = mouth_avg - eye_avg
    x = eye_to_eye - rot90(eye_to_mouth)
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = rot90(x)
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    quad_for_coeffs = quad[[0, 3, 2, 1]]
    return torch.from_numpy(quad_for_coeffs).float()


def get_face_align_matrix_celebm(face_pts: torch.Tensor, target_shape: Tuple[int, int]):
    face_pts = torch.stack([get_quad(pts) for pts in face_pts], dim=0).to(face_pts)
    assert target_shape[0] == target_shape[1]
    target_size = target_shape[0]
    target_pts = torch.as_tensor(
        [[0, 0], [target_size, 0], [target_size, target_size], [0, target_size]]
    ).to(face_pts)
    if target_pts.dim() == 2:
        target_pts = target_pts.unsqueeze(0)
    if target_pts.size(0) == 1:
        target_pts = target_pts.broadcast_to(face_pts.shape)
    assert target_pts.shape == face_pts.shape
    return get_similarity_transform_matrix(face_pts, target_pts)


@functools.lru_cache(maxsize=128)
def _meshgrid(h, w) -> Tuple[torch.Tensor, torch.Tensor]:
    yy, xx = torch.meshgrid(torch.arange(h).float(), torch.arange(w).float(), indexing="ij")
    return yy + 0.5, xx + 0.5


def _forge_grid(batch_size: int, device: torch.device, output_shape: Tuple[int, int], fn: Callable[[torch.Tensor], torch.Tensor]):
    h, w, *_ = output_shape
    yy, xx = _meshgrid(h, w)
    yy = yy.unsqueeze(0).broadcast_to(batch_size, h, w).to(device)
    xx = xx.unsqueeze(0).broadcast_to(batch_size, h, w).to(device)
    in_xxyy = torch.stack([xx, yy], dim=-1).reshape([batch_size, h * w, 2])
    out_xxyy: torch.Tensor = fn(in_xxyy)
    return out_xxyy.reshape(batch_size, h, w, 2)


def _safe_arctanh(x: torch.Tensor, eps: float = 0.001) -> torch.Tensor:
    return torch.clamp(x, -1 + eps, 1 - eps).arctanh()


def inverted_tanh_warp_transform(coords: torch.Tensor, matrix: torch.Tensor, warp_factor: float, warped_shape: Tuple[int, int]):
    h, w, *_ = warped_shape
    w_h = torch.tensor([[w, h]]).to(coords)
    if warp_factor > 0:
        coords = coords / w_h * 2 - 1
        nl_part1 = coords > 1.0 - warp_factor
        nl_part2 = coords < -1.0 + warp_factor
        ret_nl_part1 = _safe_arctanh((coords - 1.0 + warp_factor) / warp_factor) * warp_factor + 1.0 - warp_factor
        ret_nl_part2 = _safe_arctanh((coords + 1.0 - warp_factor) / warp_factor) * warp_factor - 1.0 + warp_factor
        coords = torch.where(nl_part1, ret_nl_part1, torch.where(nl_part2, ret_nl_part2, coords))
        coords = (coords + 1) / 2 * w_h
    coords_homo = torch.cat([coords, torch.ones_like(coords[:, :, [0]])], dim=-1)
    inv_matrix = torch.linalg.inv(matrix)
    coords_homo = torch.bmm(coords_homo, inv_matrix.permute(0, 2, 1))
    return coords_homo[:, :, :2] / coords_homo[:, :, [2, 2]]


def tanh_warp_transform(coords: torch.Tensor, matrix: torch.Tensor, warp_factor: float, warped_shape: Tuple[int, int]):
    h, w, *_ = warped_shape
    w_h = torch.tensor([[w, h]]).to(coords)
    coords_homo = torch.cat([coords, torch.ones_like(coords[:, :, [0]])], dim=-1)
    coords_homo = torch.bmm(coords_homo, matrix.transpose(2, 1))
    coords = coords_homo[:, :, :2] / coords_homo[:, :, [2, 2]]
    if warp_factor > 0:
        coords = coords / w_h * 2 - 1
        nl_part1 = coords > 1.0 - warp_factor
        nl_part2 = coords < -1.0 + warp_factor
        ret_nl_part1 = torch.tanh((coords - 1.0 + warp_factor) / warp_factor) * warp_factor + 1.0 - warp_factor
        ret_nl_part2 = torch.tanh((coords + 1.0 - warp_factor) / warp_factor) * warp_factor - 1.0 + warp_factor
        coords = torch.where(nl_part1, ret_nl_part1, torch.where(nl_part2, ret_nl_part2, coords))
        coords = (coords + 1) / 2 * w_h
    return coords


def make_tanh_warp_grid(matrix: torch.Tensor, warp_factor: float, warped_shape: Tuple[int, int], orig_shape: Tuple[int, int]):
    orig_h, orig_w, *_ = orig_shape
    w_h = torch.tensor([orig_w, orig_h]).to(matrix).reshape(1, 1, 1, 2)
    return _forge_grid(
        matrix.size(0),
        matrix.device,
        warped_shape,
        functools.partial(inverted_tanh_warp_transform, matrix=matrix, warp_factor=warp_factor, warped_shape=warped_shape),
    ) / w_h * 2 - 1


def make_inverted_tanh_warp_grid(
    matrix: torch.Tensor, warp_factor: float, warped_shape: Tuple[int, int], orig_shape: Tuple[int, int]
):
    h, w, *_ = warped_shape
    w_h = torch.tensor([w, h]).to(matrix).reshape(1, 1, 1, 2)
    return _forge_grid(
        matrix.size(0),
        matrix.device,
        orig_shape,
        functools.partial(tanh_warp_transform, matrix=matrix, warp_factor=warp_factor, warped_shape=warped_shape),
    ) / w_h * 2 - 1


pretrain_settings = {
    "lapa/448": {
        "url": [
            "https://github.com/FacePerceiver/facer/releases/download/models-v1/face_parsing.farl.lapa.main_ema_136500_jit191.pt",
        ],
        "matrix_src_tag": "points",
        "get_matrix_fn": functools.partial(get_face_align_matrix, target_shape=(448, 448), target_face_scale=1.0),
        "get_grid_fn": functools.partial(make_tanh_warp_grid, warp_factor=0.8, warped_shape=(448, 448)),
        "get_inv_grid_fn": functools.partial(make_inverted_tanh_warp_grid, warp_factor=0.8, warped_shape=(448, 448)),
        "label_names": ["background", "face", "rb", "lb", "re", "le", "nose", "ulip", "imouth", "llip", "hair"],
    },
    "celebm/448": {
        "url": [
            "https://github.com/FacePerceiver/facer/releases/download/models-v1/face_parsing.farl.celebm.main_ema_181500_jit.pt",
        ],
        "matrix_src_tag": "points",
        "get_matrix_fn": functools.partial(get_face_align_matrix_celebm, target_shape=(448, 448)),
        "get_grid_fn": functools.partial(make_tanh_warp_grid, warp_factor=0, warped_shape=(448, 448)),
        "get_inv_grid_fn": functools.partial(make_inverted_tanh_warp_grid, warp_factor=0, warped_shape=(448, 448)),
        "label_names": [
            "background",
            "neck",
            "face",
            "cloth",
            "rr",
            "lr",
            "rb",
            "lb",
            "re",
            "le",
            "nose",
            "imouth",
            "llip",
            "ulip",
            "hair",
            "eyeg",
            "hat",
            "earr",
            "neck_l",
        ],
    },
}


class FaRLFaceParser(FaceParser):
    def __init__(self, conf_name: Optional[str] = None, model_path: Optional[str] = None, device=None) -> None:
        super().__init__()
        if conf_name is None:
            conf_name = "lapa/448"
        if model_path is None:
            model_path = pretrain_settings[conf_name]["url"]
        self.conf_name = conf_name
        self.net = download_jit(model_path, map_location=device)
        self.eval()
        self.device = device
        self.setting = pretrain_settings[conf_name]
        self.label_names = self.setting["label_names"]

    def get_warp_grid(self, images: torch.Tensor, matrix_src):
        _, _, h, w = images.shape
        matrix = self.setting["get_matrix_fn"](matrix_src)
        grid = self.setting["get_grid_fn"](matrix=matrix, orig_shape=(h, w))
        inv_grid = self.setting["get_inv_grid_fn"](matrix=matrix, orig_shape=(h, w))
        return grid, inv_grid

    def warp_images(self, images: torch.Tensor, data: Dict[str, Any]):
        simages = self.unify_image_dtype(images)
        simages = simages[data["image_ids"]]
        matrix_src = data[self.setting["matrix_src_tag"]]
        grid, inv_grid = self.get_warp_grid(simages, matrix_src)
        w_images = F.grid_sample(simages, grid, mode="bilinear", align_corners=False)
        return w_images, grid, inv_grid

    def decode_image_to_cv2(self, images: torch.Tensor):
        assert images.ndim == 4
        assert images.shape[1] == 3
        images = images.permute(0, 2, 3, 1).cpu().numpy() * 255
        images = images.astype(np.uint8)
        return images

    def unify_image_dtype(self, images: Union[torch.Tensor, np.ndarray, list]):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        elif isinstance(images, torch.Tensor):
            pass
        elif isinstance(images, list):
            assert len(images) > 0
            first_image = images[0]
            if isinstance(first_image, np.ndarray):
                images = [torch.from_numpy(image).permute(2, 0, 1) for image in images]
                images = torch.stack(images)
            elif isinstance(first_image, torch.Tensor):
                images = torch.stack(images)
            else:
                raise ValueError(f"Unsupported image type: {type(first_image)}")
        else:
            raise ValueError(f"Unsupported image type: {type(images)}")
        assert images.ndim == 4
        assert images.shape[1] == 3
        max_val = images.max()
        if max_val <= 1:
            assert images.dtype == torch.float32 or images.dtype == torch.float16
        elif max_val <= 255:
            assert images.dtype == torch.uint8
            images = images.float() / 255.0
        else:
            raise ValueError(f"Unsupported image type: {images.dtype}")
        if images.device != self.device:
            images = images.to(device=self.device)
        return images

    @torch.no_grad()
    @torch.inference_mode()
    def forward(self, images: torch.Tensor, data: Dict[str, Any]):
        w_images, grid, inv_grid = self.warp_images(images, data)
        w_seg_logits = self.forward_warped(w_images, return_preds=False)
        seg_logits = F.grid_sample(w_seg_logits, inv_grid, mode="bilinear", align_corners=False)
        data["seg"] = {"logits": seg_logits, "label_names": self.label_names}
        return data

    def logits2predictions(self, logits: torch.Tensor):
        return logits.argmax(dim=1)

    @torch.no_grad()
    @torch.inference_mode()
    def forward_warped(self, images: torch.Tensor, return_preds: bool = True):
        images = self.unify_image_dtype(images)
        seg_logits, _ = self.net(images)
        if return_preds:
            seg_preds = self.logits2predictions(seg_logits)
            return seg_logits, seg_preds, self.label_names
        return seg_logits
