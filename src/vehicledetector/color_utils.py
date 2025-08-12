from __future__ import annotations
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any


def crop_and_resize(frame: np.ndarray, bbox_xyxy: Tuple[int, int, int, int], max_side: int = 256) -> np.ndarray:
    x1, y1, x2, y2 = bbox_xyxy
    h, w = frame.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(w - 1, x2); y2 = min(h - 1, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return crop
    ch, cw = crop.shape[:2]
    # Compute scale to limit the largest side to max_side (downscale only)
    longest = max(ch, cw)
    if longest <= 0:
        return crop
    scale = min(max_side / float(longest), 1.0)
    if scale < 1.0:
        # Ensure target size is at least 1x1 to avoid OpenCV assertion failures on extreme aspect ratios
        new_w = max(1, int(round(cw * scale)))
        new_h = max(1, int(round(ch * scale)))
        # Only resize if it actually changes dimensions
        if new_w != cw or new_h != ch:
            crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return crop


def white_mask_lab(crop_bgr: np.ndarray, lab_cfg: Dict[str, Any], erosion: int = 0) -> np.ndarray:
    # OpenCV Lab is L in [0,255], a/b in [0,255] with 128==0
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    L_min = int(lab_cfg.get("L_min", 60))
    L_max = int(lab_cfg.get("L_max", 255))
    a_min = 128 + int(lab_cfg.get("a_min", -10))
    a_max = 128 + int(lab_cfg.get("a_max", 10))
    b_min = 128 + int(lab_cfg.get("b_min", -10))
    b_max = 128 + int(lab_cfg.get("b_max", 20))
    lower = np.array([
        int(np.clip(L_min, 0, 255)),
        int(np.clip(a_min, 0, 255)),
        int(np.clip(b_min, 0, 255)),
    ], dtype=np.uint8)
    upper = np.array([
        int(np.clip(L_max, 0, 255)),
        int(np.clip(a_max, 0, 255)),
        int(np.clip(b_max, 0, 255)),
    ], dtype=np.uint8)
    mask = cv2.inRange(lab, lower, upper)
    if erosion > 0:
        kernel = np.ones((erosion, erosion), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
    return mask


def white_mask_hsv(crop_bgr: np.ndarray, hsv_cfg: Dict[str, Any], erosion: int = 0) -> np.ndarray:
    # Heuristic: white has high V and low S, hue is unreliable
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    S_max = int(hsv_cfg.get("S_max", 60))
    V_min = int(hsv_cfg.get("V_min", 180))
    V_max = int(hsv_cfg.get("V_max", 255))
    # Ignore H by taking full range
    lower = np.array([0, 0, int(np.clip(V_min, 0, 255))], dtype=np.uint8)
    upper = np.array([179, int(np.clip(S_max, 0, 255)), int(np.clip(V_max, 0, 255))], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    if erosion > 0:
        kernel = np.ones((erosion, erosion), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
    return mask


def check_white(frame: np.ndarray, bbox_xyxy: Tuple[int, int, int, int], cfg: Dict[str, Any]) -> Tuple[bool, float, Optional[np.ndarray], Optional[np.ndarray]]:
    # Returns: (is_white, white_ratio, crop_bgr, mask)
    min_area = int(cfg.get("min_box_area", 1000))
    mode = str(cfg.get("mode", "lab")).lower()
    ratio_thr = float(cfg.get("white_ratio_threshold", 0.25))
    erosion = int(cfg.get("erosion", 0))

    x1, y1, x2, y2 = bbox_xyxy
    if (x2 - x1) * (y2 - y1) < min_area:
        return False, 0.0, None, None
    crop = crop_and_resize(frame, bbox_xyxy)
    if crop.size == 0:
        return False, 0.0, None, None

    if mode == "hsv":
        mask = white_mask_hsv(crop, cfg.get("hsv", {}), erosion)
    else:  # default to Lab
        mask = white_mask_lab(crop, cfg.get("lab", {}), erosion)
    white_ratio = float((mask > 0).mean()) if mask.size else 0.0
    return (white_ratio >= ratio_thr), white_ratio, crop, mask


def is_pearl_white(frame: np.ndarray, bbox_xyxy: Tuple[int, int, int, int], lab_range: dict, min_area: int = 1000, erosion: int = 0) -> bool:
    # Back-compat path using Lab mode and default threshold 0.25
    cfg = {
        "mode": "lab",
        "lab": lab_range or {},
        "min_box_area": min_area,
        "erosion": erosion,
        "white_ratio_threshold": 0.25,
    }
    ok, _, _, _ = check_white(frame, bbox_xyxy, cfg)
    return ok
