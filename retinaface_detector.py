from typing import Dict, List, Optional, Tuple
from itertools import product as product
from math import ceil
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models._utils as _utils


class FaceDetector(nn.Module):
    pass


pretrained_urls = {
    "mobilenet": "https://github.com/elliottzheng/face-detection/releases/download/0.0.1/mobilenet0.25_Final.pth",
    "resnet50": "https://github.com/elliottzheng/face-detection/releases/download/0.0.1/Resnet50_Final.pth",
}


def conv_bn(inp, oup, stride=1, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )


def conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


def conv_dw(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if out_channel <= 64:
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel // 2, stride=1)
        self.conv5X5_1 = conv_bn(in_channel, out_channel // 4, stride=1, leaky=leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)
        self.conv7X7_2 = conv_bn(out_channel // 4, out_channel // 4, stride=1, leaky=leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)
        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)
        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)
        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        leaky = 0
        if out_channels <= 64:
            leaky = 0.1
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride=1, leaky=leaky)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride=1, leaky=leaky)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride=1, leaky=leaky)
        self.merge1 = conv_bn(out_channels, out_channels, leaky=leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky=leaky)

    def forward(self, input):
        input = list(input.values())
        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])
        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)
        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)
        return [output1, output2, output3]


class MobileNetV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(
            conv_bn(3, 8, 2, leaky=0.1),
            conv_dw(8, 16, 1),
            conv_dw(16, 32, 2),
            conv_dw(32, 32, 1),
            conv_dw(32, 64, 2),
            conv_dw(64, 64, 1),
        )
        self.stage2 = nn.Sequential(
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 128, 1),
            conv_dw(128, 128, 1),
            conv_dw(128, 128, 1),
            conv_dw(128, 128, 1),
        )
        self.stage3 = nn.Sequential(
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
        )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x


class ClassHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super().__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors * 2, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super().__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors * 4, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super().__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors * 10, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 10)


class RetinaFace(nn.Module):
    def __init__(self, cfg=None, phase="train"):
        super().__init__()
        self.phase = phase
        backbone = None
        if cfg["name"] == "mobilenet0.25":
            backbone = MobileNetV1()
        elif cfg["name"] == "Resnet50":
            import torchvision.models as models
            backbone = models.resnet50(pretrained=cfg["pretrain"])
        self.body = _utils.IntermediateLayerGetter(backbone, cfg["return_layers"])
        in_channels_stage2 = cfg["in_channel"]
        in_channels_list = [in_channels_stage2 * 2, in_channels_stage2 * 4, in_channels_stage2 * 8]
        out_channels = cfg["out_channel"]
        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)
        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg["out_channel"])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg["out_channel"])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg["out_channel"])

    def _make_class_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        classhead = nn.ModuleList()
        for _ in range(fpn_num):
            classhead.append(ClassHead(inchannels, anchor_num))
        return classhead

    def _make_bbox_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        bboxhead = nn.ModuleList()
        for _ in range(fpn_num):
            bboxhead.append(BboxHead(inchannels, anchor_num))
        return bboxhead

    def _make_landmark_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        landmarkhead = nn.ModuleList()
        for _ in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels, anchor_num))
        return landmarkhead

    def forward(self, inputs):
        out = self.body(inputs)
        fpn = self.fpn(out)
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]
        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)
        if self.phase == "train":
            return bbox_regressions, classifications, ldm_regressions
        return bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions


def decode(loc: torch.Tensor, priors: torch.Tensor, variances: Tuple[float, float]) -> torch.Tensor:
    boxes = torch.cat(
        (
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1]),
        ),
        1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landm(pre: torch.Tensor, priors: torch.Tensor, variances: Tuple[float, float]) -> torch.Tensor:
    landms = torch.cat(
        (
            priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
        ),
        dim=1,
    )
    return landms


def nms(dets: torch.Tensor, thresh: float) -> List[int]:
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = torch.flip(scores.argsort(), [0])
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])
        w = torch.maximum(torch.tensor(0.0).to(dets), xx2 - xx1 + 1)
        h = torch.maximum(torch.tensor(0.0).to(dets), yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = torch.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


class PriorBox:
    def __init__(self, cfg: dict, image_size: Tuple[int, int]):
        self.min_sizes = cfg["min_sizes"]
        self.steps = cfg["steps"]
        self.clip = cfg["clip"]
        self.image_size = image_size
        self.feature_maps = [
            [ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)]
            for step in self.steps
        ]

    def generate_anchors(self, device) -> torch.Tensor:
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
        output = torch.tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output.to(device=device)


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

cfg_re50 = {
    "name": "Resnet50",
    "min_sizes": [[16, 32], [64, 128], [256, 512]],
    "steps": [8, 16, 32],
    "variance": [0.1, 0.2],
    "clip": False,
    "loc_weight": 2.0,
    "gpu_train": True,
    "batch_size": 24,
    "ngpu": 4,
    "epoch": 100,
    "decay1": 70,
    "decay2": 90,
    "image_size": 840,
    "pretrain": False,
    "return_layers": {"layer2": 1, "layer3": 2, "layer4": 3},
    "in_channel": 256,
    "out_channel": 256,
}


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    assert len(used_pretrained_keys) > 0
    return True


def remove_prefix(state_dict, prefix):
    def f(x):
        return x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu, network: str):
    def _torch_load(path, map_location):
        try:
            return torch.load(path, map_location=map_location, weights_only=True)
        except TypeError:
            return torch.load(path, map_location=map_location)

    if pretrained_path is None:
        url = pretrained_urls[network]
        if load_to_cpu:
            pretrained_dict = torch.utils.model_zoo.load_url(
                url, map_location=lambda storage, loc: storage
            )
        else:
            device = torch.cuda.current_device()
            pretrained_dict = torch.utils.model_zoo.load_url(
                url, map_location=lambda storage, loc: storage.cuda(device)
            )
    else:
        if load_to_cpu:
            pretrained_dict = _torch_load(pretrained_path, map_location=lambda storage, loc: storage)
        else:
            device = torch.cuda.current_device()
            pretrained_dict = _torch_load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict["state_dict"], "module.")
    else:
        pretrained_dict = remove_prefix(pretrained_dict, "module.")
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def load_net(model_path, network="mobilenet", device=None):
    if network == "mobilenet":
        cfg = cfg_mnet
    elif network == "resnet50":
        cfg = cfg_re50
    else:
        raise NotImplementedError(network)
    if device is None:
        raise RuntimeError("CUDA device is required")
    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError("CUDA device is required")
    net = RetinaFace(cfg=cfg, phase="test")
    net = load_model(net, model_path, False, network=network)
    net.eval()
    cudnn.benchmark = True
    net = net.to(device)
    return net


def parse_det(det: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
    landmarks = det[5:].reshape(5, 2)
    box = det[:4]
    score = det[4]
    return box, landmarks, score.item()


def post_process(
    loc: torch.Tensor,
    conf: torch.Tensor,
    landms: torch.Tensor,
    prior_data: torch.Tensor,
    cfg: dict,
    scale: float,
    scale1: float,
    resize,
    confidence_threshold,
    top_k,
    nms_threshold,
    keep_top_k,
):
    boxes = decode(loc, prior_data, cfg["variance"])
    boxes = boxes * scale / resize
    scores = conf[:, 1]
    landms_copy = decode_landm(landms, prior_data, cfg["variance"])
    landms_copy = landms_copy * scale1 / resize
    inds = torch.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landms_copy = landms_copy[inds]
    scores = scores[inds]
    order = torch.flip(scores.argsort(), [0])[:top_k]
    boxes = boxes[order]
    landms_copy = landms_copy[order]
    scores = scores[order]
    dets = torch.hstack((boxes, scores.unsqueeze(-1))).to(dtype=torch.float32, copy=False)
    keep = nms(dets, nms_threshold)
    dets = dets[keep, :]
    landms_copy = landms_copy[keep]
    dets = dets[:keep_top_k, :]
    landms_copy = landms_copy[:keep_top_k, :]
    dets = torch.cat((dets, landms_copy), dim=1)
    dets = sorted(dets, key=lambda x: x[4], reverse=True)
    dets = [parse_det(x) for x in dets]
    return dets


def batch_detect(net: nn.Module, images: torch.Tensor, threshold: float = 0.5):
    confidence_threshold = threshold
    cfg = cfg_mnet
    top_k = 5000
    nms_threshold = 0.4
    keep_top_k = 750
    resize = 1
    img = images.float()
    mean = torch.as_tensor([104, 117, 123], dtype=img.dtype, device=img.device).view(1, 3, 1, 1)
    img -= mean
    _, _, im_height, im_width = img.shape
    scale = torch.as_tensor([im_width, im_height, im_width, im_height], dtype=img.dtype, device=img.device)
    scale = scale.to(img.device)
    loc, conf, landms = net(img)
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    prior_data = priorbox.generate_anchors(device=img.device)
    scale1 = torch.as_tensor(
        [
            img.shape[3],
            img.shape[2],
            img.shape[3],
            img.shape[2],
            img.shape[3],
            img.shape[2],
            img.shape[3],
            img.shape[2],
            img.shape[3],
            img.shape[2],
        ],
        dtype=img.dtype,
        device=img.device,
    )
    scale1 = scale1.to(img.device)
    all_dets = [
        post_process(
            loc_i,
            conf_i,
            landms_i,
            prior_data,
            cfg,
            scale,
            scale1,
            resize,
            confidence_threshold,
            top_k,
            nms_threshold,
            keep_top_k,
        )
        for loc_i, conf_i, landms_i in zip(loc, conf, landms)
    ]
    rects = []
    points = []
    scores = []
    image_ids = []
    for image_id, faces_in_one_image in enumerate(all_dets):
        for rect, landmarks, score in faces_in_one_image:
            rects.append(rect)
            points.append(landmarks)
            scores.append(score)
            image_ids.append(image_id)
    if len(rects) == 0:
        return {
            "rects": torch.Tensor().to(img.device),
            "points": torch.Tensor().to(img.device),
            "scores": torch.Tensor().to(img.device),
            "image_ids": torch.Tensor().to(img.device),
        }
    return {
        "rects": torch.stack(rects, dim=0).to(img.device),
        "points": torch.stack(points, dim=0).to(img.device),
        "scores": torch.tensor(scores).to(img.device),
        "image_ids": torch.tensor(image_ids).to(img.device),
    }


class RetinaFaceDetector(FaceDetector):
    def __init__(
        self,
        conf_name: Optional[str] = None,
        model_path: Optional[str] = None,
        threshold=0.8,
        device=None,
    ) -> None:
        super().__init__()
        if device is None:
            raise RuntimeError("CUDA device is required")
        if conf_name is None:
            conf_name = "mobilenet"
        self.net = load_net(model_path, conf_name, device=device)
        self.threshold = threshold
        self.eval()

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        return batch_detect(self.net, images.clone(), threshold=self.threshold)
