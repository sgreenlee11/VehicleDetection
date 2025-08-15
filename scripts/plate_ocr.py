import argparse
import os
from pathlib import Path
import cv2
import numpy as np
from rich import print

# Optional: OpenCV DNN Super Resolution (requires opencv-contrib-python)
try:
    from cv2 import dnn_superres  # type: ignore
    HAS_DNN_SR = True
except Exception:
    HAS_DNN_SR = False
# Optional: Real-ESRGAN for super-resolution
try:
    from realesrgan import RealESRGANer  # type: ignore
    from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore
    HAS_SR = True
except Exception:
    HAS_SR = False

# Try import easyocr; fallback to pytesseract if present
OCR_BACKENDS = []
try:
    import easyocr  # type: ignore
    OCR_BACKENDS.append('easyocr')
except Exception:
    pass
try:
    import pytesseract  # type: ignore
    OCR_BACKENDS.append('pytesseract')
except Exception:
    pass

if not OCR_BACKENDS:
    raise SystemExit("No OCR backend available. Install easyocr or pytesseract.")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def deskew(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if coords.size == 0:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def unsharp_mask(img: np.ndarray, ksize: int = 0, sigma: float = 0.0, amount: float = 1.0, thresh: int = 0) -> np.ndarray:
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    if ksize <= 0:
        # Use Gaussian blur with auto sigma
        blur = cv2.GaussianBlur(gray, (0, 0), max(1.0, sigma or 1.0))
    else:
        blur = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
    sharp = cv2.addWeighted(gray, 1 + amount, blur, -amount, 0)
    if thresh > 0:
        low_contrast_mask = np.absolute(gray - blur) < thresh
        np.copyto(sharp, gray, where=low_contrast_mask)
    return sharp


def adjust_gamma(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    invGamma = 1.0 / max(gamma, 1e-6)
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def try_warp_plate(img: np.ndarray) -> np.ndarray | None:
    # Basic 4-point contour detection to rectify a plate-like rectangle
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
            if 1.8 <= ar <= 6.0:  # plate-ish aspect range
                if area > best_area:
                    best = approx
                    best_area = area
    if best is None:
        return None
    pts = best.reshape(4, 2).astype(np.float32)
    # Order points: top-left, top-right, bottom-right, bottom-left
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    # Set target size with reasonable aspect ratio
    widthA = np.linalg.norm(ordered[2] - ordered[3])
    widthB = np.linalg.norm(ordered[1] - ordered[0])
    maxW = int(max(widthA, widthB))
    heightA = np.linalg.norm(ordered[1] - ordered[2])
    heightB = np.linalg.norm(ordered[0] - ordered[3])
    maxH = int(max(heightA, heightB))
    # enforce a standard plate aspect ~3:1, but keep enough height
    target_w = max(maxW, 200)
    target_h = max(int(target_w / 3), maxH)
    dst = np.array([[0,0],[target_w-1,0],[target_w-1,target_h-1],[0,target_h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(img, M, (target_w, target_h), flags=cv2.INTER_CUBIC)
    return warped


def enhance_variants(img: np.ndarray, upscales: list[int] | None = None) -> list[tuple[str, np.ndarray]]:
    variants: list[tuple[str, np.ndarray]] = []

    # Base
    variants.append(("base", img))

    # Grayscale + CLAHE
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_clahe = clahe.apply(gray)
    variants.append(("gray_clahe", gray_clahe))

    # Bilateral filter + adaptive threshold
    bl = cv2.bilateralFilter(gray, 9, 75, 75)
    adt = cv2.adaptiveThreshold(bl, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    variants.append(("bilateral_adapt", adt))

    # Morphological opening to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opened = cv2.morphologyEx(gray_clahe, cv2.MORPH_OPEN, kernel, iterations=1)
    variants.append(("opened", opened))

    # Sharpening
    sharpen_kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    sharp = cv2.filter2D(gray_clahe, -1, sharpen_kernel)
    variants.append(("sharpen", sharp))

    # Inverted binary for OCR variants
    _, th = cv2.threshold(gray_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu", th))
    variants.append(("otsu_inv", cv2.bitwise_not(th)))

    # Gamma variants
    gamma_vals = [0.8, 1.2, 1.6]
    for g in gamma_vals:
        gv = adjust_gamma(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), g)
        variants.append((f"gamma_{g}", gv))

    # Unsharp mask on CLAHE gray
    us = unsharp_mask(gray_clahe, sigma=1.0, amount=1.0, thresh=3)
    variants.append(("unsharp", us))

    # Resize up to help OCR (multi-scale, multiple interpolations)
    if not upscales:
        upscales = [2]
    interps = [(cv2.INTER_CUBIC, "cubic"), (cv2.INTER_LANCZOS4, "lanczos")]
    ups = []
    for name, v in variants:
        for s in upscales:
            for interp, iname in interps:
                up = cv2.resize(v, None, fx=float(s), fy=float(s), interpolation=interp)
                ups.append((f"{name}_x{s}_{iname}", up))
    variants.extend(ups)

    # Ensure all are 3-channel for easyocr; keep single channel for pytesseract
    return variants


def sr_upscale(img: np.ndarray, scale: int = 4, tile: int = 0) -> np.ndarray:
    if not HAS_SR:
        raise RuntimeError("Super-resolution library not available. Install 'realesrgan'.")
    # Define RRDBNet backbone for x4 model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path=None,  # allow library to auto-download default weights
        model=model,
        tile=tile,        # e.g., 256 for low-memory environments
        tile_pad=10,
        pre_pad=0,
        half=False,
    )
    # realesrgan expects RGB uint8
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out, _ = upsampler.enhance(rgb, outscale=scale)
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return bgr


def dnn_sr_upscale(img: np.ndarray, model_path: str, algo: str, scale: int) -> np.ndarray:
    if not HAS_DNN_SR:
        raise RuntimeError("OpenCV dnn_superres not available. Install opencv-contrib-python.")
    sr = dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel(algo.lower(), int(scale))
    # Accept gray or BGR
    if img.ndim == 2:
        src = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        src = img
    out = sr.upsample(src)
    return out


def ocr_easyocr(img: np.ndarray, languages: list[str]) -> list[str]:
    reader = easyocr.Reader(languages, gpu=False)
    result = reader.readtext(img)
    texts = [text for _, text, conf in result if (isinstance(conf, (int, float)) and conf >= 0.2) or not isinstance(conf, (int, float))]
    return texts


def ocr_pytesseract(img: np.ndarray) -> str:
    config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    if len(img.shape) == 2:
        proc = img
    else:
        proc = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return pytesseract.image_to_string(proc, config=config).strip()


def score_plate(candidate: str) -> int:
    # Simple heuristic: count alnum chars, penalize non-plate chars
    candidate = candidate.strip().upper()
    alnum = sum(c.isalnum() for c in candidate)
    bad = sum(c in {" ","-","_","/","\\","|","!","?","*","."} for c in candidate)
    return alnum * 2 - bad


def main():
    ap = argparse.ArgumentParser(description="Try OCR on a license plate crop with multiple preprocess passes")
    ap.add_argument("image", type=str, help="Path to cropped plate image")
    ap.add_argument("--langs", nargs="*", default=["en"], help="EasyOCR language codes, e.g. en, en+ru")
    ap.add_argument("--save-dir", type=str, default="outputs/plate-ocr-debug", help="Directory to store preprocessed variants")
    ap.add_argument("--sr-scale", type=int, default=0, help="Optional Real-ESRGAN upscale (2-4). 0 disables SR.")
    ap.add_argument("--sr-tile", type=int, default=0, help="Tile size for SR (0 disables tiling). Use 256/512 if low memory.")
    ap.add_argument("--upscales", type=str, default="2,3,4", help="Comma-separated integer scales to try with classic resize.")
    ap.add_argument("--auto-warp", action="store_true", help="Try perspective rectification of a plate-like rectangle.")
    # OpenCV DNN SR
    ap.add_argument("--dnn-sr-model", type=str, default="", help="Path to an OpenCV dnn_superres .pb model (EDSR/ESPCN/FSRCNN/LapSRN)")
    ap.add_argument("--dnn-sr-algo", type=str, default="edsr", help="dnn_superres algorithm: edsr|espcn|fsrcnn|lapsrn")
    ap.add_argument("--dnn-sr-scale", type=int, default=0, help="Scale for dnn_superres model (2|3|4)")
    # Candidate filtering
    ap.add_argument("--plate-len", type=int, default=0, help="Expected exact plate length (e.g., 6). If set and no custom --pattern, we enforce {len}.")
    ap.add_argument("--min-len", type=int, default=4, help="Minimum plate length to favor")
    ap.add_argument("--max-len", type=int, default=8, help="Maximum plate length to favor")
    ap.add_argument("--pattern", type=str, default=r"^[A-Z0-9]{4,8}$", help="Regex the ideal plate should match")
    ap.add_argument("--aggregate-easyocr", action="store_true", help="Aggregate EasyOCR boxes left-to-right into one line plate candidate")
    ap.add_argument("--tess-psms", type=str, default="7,8", help="Comma-separated Tesseract PSM modes to try (e.g., 7,8,6)")
    args = ap.parse_args()

    in_path = Path(args.image)
    if not in_path.exists():
        raise SystemExit(f"Image not found: {in_path}")

    out_dir = Path(args.save_dir)
    ensure_dir(out_dir)

    img = cv2.imread(str(in_path))
    if img is None:
        raise SystemExit("Failed to read image")

    # Optional super-resolution
    sr_img = None
    if args.sr_scale and args.sr_scale >= 2:
        if not HAS_SR:
            print("[yellow]Super-resolution requested but 'realesrgan' not installed.[/yellow]")
        else:
            try:
                sr_img = sr_upscale(img, scale=int(args.sr_scale), tile=int(args.sr_tile))
                print(f"[green]Applied SR x{args.sr_scale}[/green]")
            except Exception as e:
                print(f"[yellow]SR failed: {e}[/yellow]")

    # Optional OpenCV DNN SR
    if sr_img is None and args.dnn_sr_model and args.dnn_sr_scale:
        if not HAS_DNN_SR:
            print("[yellow]OpenCV dnn_superres not available. Install opencv-contrib-python.[/yellow]")
        else:
            try:
                sr_img = dnn_sr_upscale(img, args.dnn_sr_model, args.dnn_sr_algo, int(args.dnn_sr_scale))
                print(f"[green]Applied dnn_superres {args.dnn_sr_algo} x{args.dnn_sr_scale}[/green]")
            except Exception as e:
                print(f"[yellow]dnn_superres failed: {e}[/yellow]")

    base_for_processing = sr_img if sr_img is not None else img

    # Optional perspective warp
    if args.auto_warp:
        warped = try_warp_plate(base_for_processing)
        if warped is not None:
            print("[green]Applied perspective warp[/green]")
            base_for_processing = warped

    # Step 1: deskew
    rot = deskew(base_for_processing)

    # Step 2: generate variants
    # Parse upscales list
    try:
        ups = [int(x) for x in str(args.upscales).split(',') if x.strip()]
        ups = [u for u in ups if 2 <= u <= 4]
        if not ups:
            ups = [2]
    except Exception:
        ups = [2]
    variants = enhance_variants(rot, upscales=ups)

    # Save variants for inspection
    base_name = in_path.stem
    if sr_img is not None:
        cv2.imwrite(str(out_dir / f"{base_name}_srx{args.sr_scale}.png"), sr_img)
    if args.auto_warp and base_for_processing is not None and base_for_processing is not img:
        cv2.imwrite(str(out_dir / f"{base_name}_warped.png"), base_for_processing)
    for name, im in variants:
        tosave = im
        if im.ndim == 2:
            tosave = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(out_dir / f"{base_name}_{name}.png"), tosave)

    # Possibly tighten pattern/length defaults based on --plate-len
    DEFAULT_PATTERN = r"^[A-Z0-9]{4,8}$"
    if args.plate_len and args.pattern == DEFAULT_PATTERN:
        args.pattern = rf"^[A-Z0-9]{{{int(args.plate_len)}}}$"
        args.min_len = int(args.plate_len)
        args.max_len = int(args.plate_len)

    # Step 3: OCR attempts
    candidates: list[tuple[str, str, str, np.ndarray]] = []  # (raw_text, source, variant_name, image)
    for name, im in variants:
        if 'easyocr' in OCR_BACKENDS:
            try:
                texts = ocr_easyocr(im, args.langs)
                for t in texts:
                    if t:
                        candidates.append((t, f"easyocr:{name}", name, im))
                # Optional aggregation of EasyOCR boxes into one line
                if args.aggregate_easyocr:
                    try:
                        import easyocr  # type: ignore
                        reader = easyocr.Reader(args.langs, gpu=False)
                        result = reader.readtext(im)
                        # result: list of (bbox(list of 4 pts), text, conf)
                        parts = []
                        for item in result:
                            if not isinstance(item, (list, tuple)) or len(item) < 3:
                                continue
                            box, text, conf = item[0], item[1], item[2]
                            if isinstance(conf, (int, float)) and conf < 0.05:
                                continue
                            if not text:
                                continue
                            # left coordinate for sorting
                            try:
                                xs = [p[0] for p in box]
                                left = float(min(xs))
                            except Exception:
                                left = 0.0
                            parts.append((left, text))
                        if parts:
                            parts.sort(key=lambda x: x[0])
                            agg = ''.join(''.join(ch for ch in t if ch.isalnum()) for _, t in parts)
                            if agg:
                                candidates.append((agg, f"easyocr-agg:{name}", name, im))
                    except Exception:
                        pass
            except Exception as e:
                print(f"[yellow]easyocr failed on {name}: {e}[/yellow]")
        if 'pytesseract' in OCR_BACKENDS:
            try:
                # Try multiple PSMs
                psms = []
                try:
                    psms = [int(x) for x in str(args.tess_psms).split(',') if x.strip()]
                except Exception:
                    psms = [7, 8]
                psms = [p for p in psms if p in {3,4,5,6,7,8,9,10,11,12,13}]
                if not psms:
                    psms = [7, 8]
                for p in psms:
                    try:
                        import pytesseract  # type: ignore
                        config = f"--psm {p} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                        proc = im if im.ndim == 2 else cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                        txt = pytesseract.image_to_string(proc, config=config).strip()
                        if txt:
                            candidates.append((txt, f"tesseract-psm{p}:{name}", name, im))
                    except Exception:
                        continue
            except Exception as e:
                print(f"[yellow]tesseract failed on {name}: {e}[/yellow]")

    if not candidates:
        print("[red]No OCR text found. Inspect saved variants for manual tuning.[/red]")
        return

    # Normalize candidates and score with regex/length bias
    import re

    def normalize(s: str) -> str:
        s = s.upper()
        s = ''.join(ch for ch in s if ch.isalnum())
        # Common OCR confusions
        trans = str.maketrans({
            'O': '0',
            'I': '1',
            'L': '1',
            'Z': '2',
            'S': '5',
            'B': '8',
            'Q': '0',
        })
        return s.translate(trans)

    normed = []
    seen = set()
    for t, src, vname, vimg in candidates:
        n = normalize(t)
        if not n:
            continue
        if (n, src, vname) in seen:
            continue
        seen.add((n, src, vname))
        normed.append((n, src, vname, vimg))

    pat = re.compile(args.pattern)

    def score_advanced(s: str) -> int:
        base = score_plate(s)
        length = len(s)
        bonus = 0
        if pat.match(s):
            bonus += 10
        if args.min_len <= length <= args.max_len:
            bonus += 5
        # Penalize too short
        if length < args.min_len:
            bonus -= (args.min_len - length) * 2
        return base + bonus

    best = max(normed, key=lambda x: score_advanced(x[0]))
    print("\n[bold]Candidates[/bold]:")
    for t, src, vname, _ in normed[:10]:
        print(f"- {t}  ({src})  score={score_advanced(t)}  via={vname}")
    print(f"\n[green]Best guess[/green]: {best[0]}  [dim]from {best[1]}[/dim] via variant {best[2]}")

    # Export the exact enhanced variant image that produced the best candidate
    best_text, best_src, best_variant_name, best_img = best
    best_out_path = out_dir / f"{base_name}_BEST_{best_variant_name}.png"
    tosave = best_img
    if best_img.ndim == 2:
        tosave = cv2.cvtColor(best_img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(best_out_path), tosave)
    print(f"[cyan]Saved best variant image:[/cyan] {best_out_path}")

    # Optional: export top-K variant images for inspection
    TOPK = 5
    sorted_normed = sorted(normed, key=lambda x: score_advanced(x[0]), reverse=True)
    for i, (t, src, vname, vimg) in enumerate(sorted_normed[:TOPK], start=1):
        p = out_dir / f"{base_name}_TOP{i}_{vname}.png"
        vv = vimg if vimg.ndim == 3 else cv2.cvtColor(vimg, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(p), vv)
    if sorted_normed:
        print(f"[cyan]Saved top {min(TOPK, len(sorted_normed))} variant images for review in {out_dir}[/cyan]")


if __name__ == "__main__":
    main()
