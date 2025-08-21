import argparse
from pathlib import Path
import cv2
import numpy as np
from rich import print
from math import cos, sin, radians

# Optional dnn_superres
try:
    from cv2 import dnn_superres  # type: ignore
    HAS_DNN_SR = True
except Exception:
    HAS_DNN_SR = False


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_img(path: Path, img: np.ndarray):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(path), img)


essential_steps = [
    "deskew", "warp", "sr", "denoise", "sharpen", "contrast", "gamma"
]


def deskew(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_inv = cv2.bitwise_not(gray)
    thr = cv2.threshold(gray_inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thr > 0))
    if coords.size == 0:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def try_warp(img: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    best = None
    best_area = 0
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, cw, ch = cv2.boundingRect(approx)
            area = cw * ch
            if area < 0.02 * w * h:
                continue
            ar = cw / max(ch, 1)
            if 1.8 <= ar <= 6.0:
                if area > best_area:
                    best = approx
                    best_area = area
    if best is None:
        return None
    pts = best.reshape(4,2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    ordered = np.zeros((4,2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    widthA = np.linalg.norm(ordered[2]-ordered[3])
    widthB = np.linalg.norm(ordered[1]-ordered[0])
    maxW = int(max(widthA, widthB))
    heightA = np.linalg.norm(ordered[1]-ordered[2])
    heightB = np.linalg.norm(ordered[0]-ordered[3])
    maxH = int(max(heightA, heightB))
    target_w = max(maxW, 300)
    target_h = max(int(target_w/3), maxH)
    dst = np.array([[0,0],[target_w-1,0],[target_w-1,target_h-1],[0,target_h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(img, M, (target_w, target_h), flags=cv2.INTER_CUBIC)
    return warped


def dnn_sr(img: np.ndarray, model_path: str, algo: str, scale: int) -> np.ndarray:
    if not HAS_DNN_SR:
        raise RuntimeError("OpenCV dnn_superres not available. Install opencv-contrib-python.")
    sr = dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel(algo.lower(), int(scale))
    if img.ndim == 2:
        src = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        src = img
    return sr.upsample(src)


# --- Motion deblurring utilities ---
def motion_psf(length: int = 9, angle_deg: float = 0.0) -> np.ndarray:
    """Create a linear motion blur PSF kernel of given length and angle.
    length: odd recommended; clamped to >= 3
    angle_deg: 0 = horizontal motion (blur along x)
    """
    L = max(3, int(length))
    if L % 2 == 0:
        L += 1
    k = np.zeros((L, L), dtype=np.float32)
    # draw a line across the center according to angle
    cx = cy = L // 2
    theta = radians(angle_deg)
    # parametric line sampling across [-1,1]
    # construct endpoints to cover kernel
    dx = cos(theta)
    dy = sin(theta)
    # number of samples proportional to L
    samples = max(L * 4, 32)
    for t in np.linspace(-1, 1, samples):
        x = cx + t * dx * (L//2)
        y = cy + t * dy * (L//2)
        xi = int(round(x))
        yi = int(round(y))
        if 0 <= yi < L and 0 <= xi < L:
            k[yi, xi] = 1.0
    s = k.sum()
    if s > 0:
        k /= s
    else:
        k[cy, :] = 1.0 / L
    return k


def wiener_deconv(img: np.ndarray, psf: np.ndarray, K: float = 0.01) -> np.ndarray:
    """Wiener deconvolution in frequency domain (per-channel)."""
    if img.ndim == 2:
        src = img.astype(np.float32) / 255.0
        out = _wiener_single(src, psf, K)
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    # color: split channels
    bgr = cv2.split(img.astype(np.float32) / 255.0)
    rec = [_wiener_single(ch, psf, K) for ch in bgr]
    out = cv2.merge([np.clip(r * 255.0, 0, 255).astype(np.uint8) for r in rec])
    return out


def _wiener_single(img_f32: np.ndarray, psf: np.ndarray, K: float) -> np.ndarray:
    h, w = img_f32.shape[:2]
    # pad PSF to image size, center it (use wrap-around shift)
    H = np.zeros((h, w), dtype=np.float32)
    kh, kw = psf.shape
    y0 = (h - kh) // 2
    x0 = (w - kw) // 2
    H[y0:y0+kh, x0:x0+kw] = psf
    H = np.fft.fft2(np.fft.ifftshift(H))
    G = np.fft.fft2(img_f32)
    H_conj = np.conj(H)
    denom = (H * H_conj) + K
    F_hat = (H_conj / denom) * G
    f_rec = np.fft.ifft2(F_hat).real
    return np.clip(f_rec, 0.0, 1.0)


def richardson_lucy(img: np.ndarray, psf: np.ndarray, iterations: int = 15) -> np.ndarray:
    """Richardson–Lucy deconvolution (grayscale, applied to luminance)."""
    # operate on luminance to avoid artifacts
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y = y.astype(np.float32) / 255.0
    psf = psf.astype(np.float32)
    eps = 1e-7
    est = y.copy()
    psf_mirror = cv2.flip(psf, -1)
    for _ in range(max(1, int(iterations))):
        conv = cv2.filter2D(est, -1, psf, borderType=cv2.BORDER_REPLICATE)
        relative_blur = y / (conv + eps)
        est *= cv2.filter2D(relative_blur, -1, psf_mirror, borderType=cv2.BORDER_REPLICATE)
        est = np.clip(est, 0.0, 1.0)
    y_rec = (est * 255.0).astype(np.uint8)
    out = cv2.merge([y_rec, cr, cb])
    return cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR)


def denoise(img: np.ndarray, h: float = 7.0) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)


def sharpen(img: np.ndarray, amount: float = 1.0) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0,0), 1.0)
    sharp = cv2.addWeighted(gray, 1 + amount, blur, -amount, 0)
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)


def contrast(img: np.ndarray, clip: float = 2.0) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    merged = cv2.merge([l2, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def gamma(img: np.ndarray, g: float = 1.2) -> np.ndarray:
    table = np.array([((i / 255.0) ** (1.0/max(g,1e-6))) * 255 for i in range(256)]).astype('uint8')
    if img.ndim == 3:
        return cv2.LUT(img, table)
    else:
        return cv2.cvtColor(cv2.LUT(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), table), cv2.COLOR_GRAY2BGR)


def focus_score(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def edge_density(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    return float(edges.mean())


# --- Edge-mask utilities for masked deblur ---
def edge_mask(
    img: np.ndarray,
    method: str = "canny",
    canny_thr1: int = 80,
    canny_thr2: int = 160,
    dilate: int = 1,
    dilate_ksize: int = 3,
    feather: int = 3,
    gamma_pow: float = 1.0,
    thresh: float | None = None,
) -> np.ndarray:
    """Return a soft edge mask in [0,1], same HxW as input.
    - method: 'canny' or 'sobel'
    - canny_thr*: used when method=canny
    - dilate: iterations of dilation with an elliptical kernel (ksize)
    - feather: Gaussian blur radius (roughly). 0 to disable
    - gamma_pow: raise mask to this power to adjust softness ( >1 makes edges sparser)
    - thresh: optional threshold (0..1) after normalization to binarize before feathering
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if method.lower() == "sobel":
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        m = mag
    else:
        edges = cv2.Canny(gray, int(canny_thr1), int(canny_thr2))
        m = edges.astype(np.float32)

    # Normalize to [0,1]
    if m.dtype != np.float32:
        m = m.astype(np.float32)
    if m.max() > 0:
        m = m / float(m.max())

    # Optional hard threshold before feathering
    if thresh is not None:
        t = float(max(0.0, min(1.0, thresh)))
        m = (m >= t).astype(np.float32)

    # Dilate to expand edge regions
    if dilate and dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1, int(dilate_ksize)), max(1, int(dilate_ksize))))
        m = cv2.dilate((m * 255).astype(np.uint8), k, iterations=int(dilate))
        m = m.astype(np.float32) / 255.0

    # Feather (Gaussian blur)
    if feather and feather > 0:
        ksz = int(feather) * 2 + 1
        m = cv2.GaussianBlur(m, (ksz, ksz), 0)

    # Gamma to modulate softness
    g = max(1e-6, float(gamma_pow))
    m = np.power(np.clip(m, 0.0, 1.0), g)
    return np.clip(m, 0.0, 1.0)


def mask_blend(base: np.ndarray, deblur: np.ndarray, mask01: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Blend deblurred image into base guided by mask (0..1), with global strength 0..1."""
    s = float(max(0.0, min(1.0, strength)))
    if base.shape[:2] != deblur.shape[:2]:
        deblur = cv2.resize(deblur, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_CUBIC)
    if mask01.shape[:2] != base.shape[:2]:
        mask01 = cv2.resize(mask01, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_LINEAR)
    m3 = np.repeat(mask01[..., None], 3, axis=2) * s
    out = base.astype(np.float32) * (1.0 - m3) + deblur.astype(np.float32) * m3
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description="Iterative plate enhancement pipeline (no OCR)")
    ap.add_argument("image", type=str, help="Path to plate crop image")
    ap.add_argument("--out", type=str, default="outputs/plate-enhance", help="Output directory for intermediates")
    ap.add_argument("--auto-warp", action="store_true", help="Try to rectify perspective")
    # Deskew variants (simple resampling) for safer outputs
    ap.add_argument("--deskew-scales", type=str, default=None, help="Comma list of scales to export from deskew (e.g., 1.5,2,3)")
    ap.add_argument(
        "--deskew-interp",
        type=str,
        default="lanczos",
        choices=["nearest", "linear", "cubic", "area", "lanczos"],
        help="Interpolation for deskew resizes",
    )
    ap.add_argument("--sr-model", type=str, default="", help="Path to dnn_superres .pb model (EDSR/ESPCN/FSRCNN/LapSRN)")
    ap.add_argument("--sr-algo", type=str, default="edsr", help="Algorithm name")
    ap.add_argument("--sr-scale", type=int, default=4, help="SR scale 2|3|4")
    # Deblur options
    ap.add_argument("--deblur-type", type=str, default="none", choices=["none", "wiener", "rl"], help="Deblur algorithm: none|wiener|rl (Richardson–Lucy)")
    ap.add_argument("--deblur-len", type=int, default=9, help="Motion PSF length (pixels)")
    ap.add_argument("--deblur-angle", type=float, default=0.0, help="Motion PSF angle in degrees (0 = horizontal)")
    ap.add_argument("--deblur-k", type=float, default=0.01, help="Wiener constant K (noise power)")
    ap.add_argument("--deblur-iters", type=int, default=15, help="Richardson–Lucy iterations")
    ap.add_argument("--deblur-auto", action="store_true", help="Auto grid-search small set of angles/lengths and pick best (focus score)")
    ap.add_argument("--deblur-fast", action="store_true", help="Faster auto grid (smaller search + fewer RL iters)")
    ap.add_argument("--deblur-grid-angles", type=str, default=None, help="Comma offsets added to --deblur-angle (e.g., -5,0,5)")
    ap.add_argument("--deblur-grid-lens", type=str, default=None, help="Comma offsets added to --deblur-len (e.g., -2,0,2)")
    ap.add_argument("--deblur-max-width", type=int, default=640, help="If image wider, temporarily downscale to this width during deblur for speed")
    # Masked deblur options
    ap.add_argument("--deblur-mask", action="store_true", help="Blend deblur only on edges (edge-masked deblur)")
    ap.add_argument("--mask-method", type=str, default="canny", choices=["canny", "sobel"], help="Edge detector for mask")
    ap.add_argument("--mask-canny-thr1", type=int, default=80, help="Canny threshold1")
    ap.add_argument("--mask-canny-thr2", type=int, default=160, help="Canny threshold2")
    ap.add_argument("--mask-dilate", type=int, default=1, help="Dilation iterations for edge mask")
    ap.add_argument("--mask-dilate-ksize", type=int, default=3, help="Dilation kernel size for edge mask")
    ap.add_argument("--mask-feather", type=int, default=3, help="Feather (Gaussian blur radius) for mask")
    ap.add_argument("--mask-gamma", type=float, default=1.0, help="Gamma pow on mask to adjust softness (>1 sparser)")
    ap.add_argument("--mask-thresh", type=float, default=None, help="Optional threshold (0..1) to binarize mask before feather")
    ap.add_argument("--mask-strength", type=float, default=1.0, help="Blend strength 0..1 for masked deblur")
    ap.add_argument("--mask-save", action="store_true", help="Save mask image for inspection")
    ap.add_argument("--denoise", type=float, default=7.0, help="fastNlMeansDenoisingColored strength (0 to skip)")
    ap.add_argument("--sharpen", type=float, default=1.0, help="Unsharp amount (0 to skip)")
    ap.add_argument("--clahe", type=float, default=2.0, help="CLAHE clip limit (0 to skip)")
    ap.add_argument("--gamma", type=float, default=1.2, help="Gamma (>1 lighten, <1 darken, 0 to skip)")
    ap.add_argument("--mosaic", action="store_true", help="Save a mosaic of key steps")
    args = ap.parse_args()

    in_path = Path(args.image)
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    img = cv2.imread(str(in_path))
    if img is None:
        raise SystemExit("Failed to read image")

    steps = []
    # 0. Original
    steps.append(("00_original", img))

    # 1. Deskew
    dsk = deskew(img)
    steps.append(("10_deskew", dsk))

    # 1b. Optional deskew scaling variants
    if args.deskew_scales:
        def interp_from_name(name: str) -> int:
            name = name.lower()
            return {
                "nearest": cv2.INTER_NEAREST,
                "linear": cv2.INTER_LINEAR,
                "cubic": cv2.INTER_CUBIC,
                "area": cv2.INTER_AREA,
                "lanczos": cv2.INTER_LANCZOS4,
            }.get(name, cv2.INTER_LANCZOS4)

        try:
            scales = [float(s.strip()) for s in args.deskew_scales.split(',') if s.strip()]
            inter = interp_from_name(args.deskew_interp)
            for sc in scales:
                if sc <= 0:
                    continue
                new_wh = (max(1, int(dsk.shape[1] * sc)), max(1, int(dsk.shape[0] * sc)))
                dsk_res = cv2.resize(dsk, new_wh, interpolation=inter)
                steps.append((f"15_deskew_x{sc:g}_{args.deskew_interp}", dsk_res))
        except Exception as e:
            print(f"[yellow]Deskew variants failed: {e}[/yellow]")

    # 2. Warp (optional)
    base = dsk
    if args.auto_warp:
        warped = try_warp(dsk)
        if warped is not None:
            steps.append(("20_warped", warped))
            base = warped

    # 3. Super-resolution (optional)
    sr_img = base
    if args.sr_model:
        try:
            sr_img = dnn_sr(base, args.sr_model, args.sr_algo, args.sr_scale)
            steps.append((f"30_sr_{args.sr_algo}x{args.sr_scale}", sr_img))
        except Exception as e:
            print(f"[yellow]SR failed: {e}[/yellow]")

    # 4. Deblur (optional, after SR)
    proc = sr_img
    if args.deblur_type != "none":
        try:
            pre_deblur = proc.copy()
            candidates: list[tuple[str, np.ndarray, float]] = []
            if args.deblur_auto:
                # Build grid around provided angle/length
                def parse_offsets(s: str | None, fallback: list[float]) -> list[float]:
                    if not s:
                        return fallback
                    try:
                        return [float(x.strip()) for x in s.split(',') if x.strip() != '']
                    except Exception:
                        return fallback

                base_ang = float(args.deblur_angle)
                base_len = int(args.deblur_len)
                if args.deblur_fast:
                    ang_offs = parse_offsets(args.deblur_grid_angles, [-5.0, 0.0, 5.0])
                    len_offs = parse_offsets(args.deblur_grid_lens, [-2.0, 0.0, 2.0])
                    if args.deblur_type == "rl":
                        # reduce RL iters in fast mode
                        args.deblur_iters = max(6, min(args.deblur_iters, 10))
                else:
                    ang_offs = parse_offsets(args.deblur_grid_angles, [-10.0, -5.0, 0.0, 5.0, 10.0])
                    len_offs = parse_offsets(args.deblur_grid_lens, [-4.0, -2.0, 0.0, 2.0, 4.0])
                angle_grid = [base_ang + d for d in ang_offs]
                len_grid = [max(3, int(base_len + d)) for d in len_offs]

                # Optional temporary downscale for speed
                deblur_src = proc
                orig_wh = (proc.shape[1], proc.shape[0])
                if args.deblur_max_width and proc.shape[1] > int(args.deblur_max_width):
                    scale = float(args.deblur_max_width) / float(proc.shape[1])
                    new_wh = (int(proc.shape[1] * scale), int(proc.shape[0] * scale))
                    deblur_src = cv2.resize(proc, new_wh, interpolation=cv2.INTER_AREA)

                print(f"[cyan]Deblur grid[/cyan]: angles={angle_grid}, lens={len_grid} (type={args.deblur_type})")
                for ang in angle_grid:
                    for ln in len_grid:
                        psf = motion_psf(ln, ang)
                        if args.deblur_type == "wiener":
                            rec_small = wiener_deconv(deblur_src, psf, K=float(args.deblur_k))
                            # resize back if we downscaled
                            if deblur_src is not proc:
                                rec = cv2.resize(rec_small, orig_wh, interpolation=cv2.INTER_CUBIC)
                            else:
                                rec = rec_small
                            tag = f"wiener_L{ln}_A{int(ang)}_K{args.deblur_k}"
                        else:
                            rec_small = richardson_lucy(deblur_src, psf, iterations=int(args.deblur_iters))
                            if deblur_src is not proc:
                                rec = cv2.resize(rec_small, orig_wh, interpolation=cv2.INTER_CUBIC)
                            else:
                                rec = rec_small
                            tag = f"rl_L{ln}_A{int(ang)}_I{args.deblur_iters}"
                        sc = focus_score(rec)
                        candidates.append((tag, rec, sc))
            else:
                psf = motion_psf(int(args.deblur_len), float(args.deblur_angle))
                # Optional temporary downscale for speed
                deblur_src = proc
                orig_wh = (proc.shape[1], proc.shape[0])
                if args.deblur_max_width and proc.shape[1] > int(args.deblur_max_width):
                    scale = float(args.deblur_max_width) / float(proc.shape[1])
                    new_wh = (int(proc.shape[1] * scale), int(proc.shape[0] * scale))
                    deblur_src = cv2.resize(proc, new_wh, interpolation=cv2.INTER_AREA)
                if args.deblur_type == "wiener":
                    rec_small = wiener_deconv(deblur_src, psf, K=float(args.deblur_k))
                    rec = cv2.resize(rec_small, orig_wh, interpolation=cv2.INTER_CUBIC) if deblur_src is not proc else rec_small
                    candidates.append((f"wiener_L{args.deblur_len}_A{int(args.deblur_angle)}_K{args.deblur_k}", rec, focus_score(rec)))
                else:
                    rec_small = richardson_lucy(deblur_src, psf, iterations=int(args.deblur_iters))
                    rec = cv2.resize(rec_small, orig_wh, interpolation=cv2.INTER_CUBIC) if deblur_src is not proc else rec_small
                    candidates.append((f"rl_L{args.deblur_len}_A{int(args.deblur_angle)}_I{args.deblur_iters}", rec, focus_score(rec)))

            # choose best by focus score, append to steps
            tag, best_rec, sc = max(candidates, key=lambda t: t[2])
            steps.append((f"35_deblur_{tag}", best_rec))

            # If requested, apply edge-masked blending and continue with blended result
            if args.deblur_mask:
                M = edge_mask(
                    pre_deblur,
                    method=args.mask_method,
                    canny_thr1=args.mask_canny_thr1,
                    canny_thr2=args.mask_canny_thr2,
                    dilate=args.mask_dilate,
                    dilate_ksize=args.mask_dilate_ksize,
                    feather=args.mask_feather,
                    gamma_pow=args.mask_gamma,
                    thresh=args.mask_thresh,
                )
                if args.mask_save:
                    m_vis = (np.clip(M, 0.0, 1.0) * 255).astype(np.uint8)
                    m_vis = cv2.cvtColor(m_vis, cv2.COLOR_GRAY2BGR)
                    steps.append(("34_edge_mask", m_vis))
                blended = mask_blend(pre_deblur, best_rec, M, strength=args.mask_strength)
                steps.append((f"36_deblur_masked_{tag}", blended))
                proc = blended
            else:
                proc = best_rec
        except Exception as e:
            print(f"[yellow]Deblur failed: {e}[/yellow]")

    # 5. Denoise
    if args.denoise and args.denoise > 0:
        proc = denoise(proc, args.denoise)
        steps.append((f"40_denoise_{args.denoise}", proc))

    # 6. Sharpen
    if args.sharpen and args.sharpen > 0:
        proc = sharpen(proc, args.sharpen)
        steps.append((f"50_sharpen_{args.sharpen}", proc))

    # 7. Contrast (CLAHE)
    if args.clahe and args.clahe > 0:
        proc = contrast(proc, args.clahe)
        steps.append((f"60_clahe_{args.clahe}", proc))

    # 8. Gamma
    if args.gamma and args.gamma > 0:
        proc = gamma(proc, args.gamma)
        steps.append((f"70_gamma_{args.gamma}", proc))

    # Save all steps and a quick metrics CSV
    rows = ["name,focus,edges,width,height"]
    for name, im in steps:
        save_img(out_dir / f"{in_path.stem}_{name}.png", im)
        rows.append(f"{name},{focus_score(im):.2f},{edge_density(im):.4f},{im.shape[1]},{im.shape[0]}")
    (out_dir / f"{in_path.stem}_metrics.csv").write_text("\n".join(rows), encoding="utf-8")

    print(f"[green]Saved {len(steps)} enhancement stages to {out_dir}[/green]")
    print(f"[cyan]Metrics CSV:[/cyan] {(out_dir / f'{in_path.stem}_metrics.csv')} ")

    # Optional mosaic
    if args.mosaic:
        try:
            tiles = []
            max_w = max(im.shape[1] for _, im in steps)
            # Build labeled thumbnails
            for name, im in steps:
                scale = max_w / max(1, im.shape[1])
                thumb = cv2.resize(im, (int(im.shape[1]*scale*0.5), int(im.shape[0]*scale*0.5)), interpolation=cv2.INTER_AREA)
                label = np.full((24, thumb.shape[1], 3), 255, dtype=np.uint8)
                cv2.putText(label, name, (5,18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                tiles.append(np.vstack([thumb, label]))

            if not tiles:
                raise RuntimeError("No tiles to mosaic")

            # Group tiles into rows of up to 3, pad each row to same width
            row_imgs = []
            row_widths = []
            for i in range(0, len(tiles), 3):
                row_tiles = tiles[i:i+3]
                # Equalize heights in the row
                max_h = max(t.shape[0] for t in row_tiles)
                norm_row_tiles = []
                for t in row_tiles:
                    if t.shape[0] != max_h:
                        pad = np.full((max_h - t.shape[0], t.shape[1], 3), 255, dtype=np.uint8)
                        t = np.vstack([t, pad])
                    norm_row_tiles.append(t)
                row = norm_row_tiles[0]
                for t in norm_row_tiles[1:]:
                    row = np.hstack([row, t])
                row_imgs.append(row)
                row_widths.append(row.shape[1])

            target_w = max(row_widths)
            # Pad rows to same width
            for idx in range(len(row_imgs)):
                r = row_imgs[idx]
                if r.shape[1] < target_w:
                    pad = np.full((r.shape[0], target_w - r.shape[1], 3), 255, dtype=np.uint8)
                    row_imgs[idx] = np.hstack([r, pad])

            mosaic = row_imgs[0]
            for r in row_imgs[1:]:
                mosaic = np.vstack([mosaic, r])
            save_img(out_dir / f"{in_path.stem}_mosaic.png", mosaic)
            print(f"[cyan]Mosaic saved:[/cyan] {(out_dir / f'{in_path.stem}_mosaic.png')} ")
        except Exception as e:
            print(f"[yellow]Mosaic failed: {e}")


if __name__ == "__main__":
    main()
