import argparse
import os
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# Optional TorchReID import
try:
    import torchreid
    HAS_TORCHREID = True
except Exception:
    HAS_TORCHREID = False


def load_images(folder: str, exts=(".jpg", ".jpeg", ".png", ".bmp")) -> List[Path]:
    p = Path(folder)
    files = []
    if not p.exists():
        return files
    for f in p.rglob('*'):
        if f.is_file() and f.suffix.lower() in exts:
            files.append(f)
    return sorted(files)


def build_transform(size: int = 256):
    return T.Compose([
        T.Resize(size),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_model(device: torch.device, arch: str, weights: str | None):
    if HAS_TORCHREID and arch.startswith('osnet'):
        # TorchReID model zoo
        model = torchreid.models.build_model(
            name=arch,
            num_classes=1,  # unused for feature extraction
            pretrained=True if not weights else False
        )
        if weights and Path(weights).exists():
            sd = torch.load(weights, map_location='cpu')
            model.load_state_dict(sd, strict=False)
        model.eval().to(device)
        return model, True
    # Fallback to torchvision ResNet50
    from torchvision.models import resnet50, ResNet50_Weights
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    # Use features before classifier by removing the final fc
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    return model, False


def extract_embeddings(model, is_reid: bool, images: List[Path], device: torch.device, size: int = 256, batch: int = 32):
    tfm = build_transform(size)
    embs = []
    with torch.inference_mode():
        for i in range(0, len(images), batch):
            batch_imgs = []
            for p in images[i:i+batch]:
                try:
                    img = Image.open(p).convert('RGB')
                except Exception:
                    continue
                batch_imgs.append(tfm(img))
            if not batch_imgs:
                continue
            x = torch.stack(batch_imgs, dim=0).to(device)
            feats = model(x)
            if not is_reid:
                # ResNet50 outputs 2048-dim; that's fine
                pass
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)
            embs.append(feats.cpu().numpy())
    if not embs:
        return np.zeros((0, 2048), dtype=np.float32)
    return np.concatenate(embs, axis=0)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: Nxd, b: Mxd -> NxM
    return a @ b.T


def main():
    ap = argparse.ArgumentParser(description='Offline vehicle re-ID similarity scorer')
    ap.add_argument('--gallery', required=True, help='Folder with reference images of the target vehicle')
    ap.add_argument('--queries', required=True, help='Folder with query images (e.g., exported still crops)')
    ap.add_argument('--arch', default='osnet_x1_0', help='Backbone: osnet_x1_0 (TorchReID) or fallback to resnet50')
    ap.add_argument('--weights', default=None, help='Optional path to model weights (TorchReID)')
    ap.add_argument('--size', type=int, default=256, help='Input size for embeddings')
    ap.add_argument('--batch', type=int, default=32, help='Batch size')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--threshold', type=float, default=0.85, help='Cosine similarity threshold to consider a match')
    ap.add_argument('--out', default='reid_matches.csv', help='CSV output with matches')
    args = ap.parse_args()

    device = torch.device(args.device)
    model, is_reid = load_model(device, args.arch, args.weights)

    gal_files = load_images(args.gallery)
    qry_files = load_images(args.queries)
    if not gal_files:
        print('No gallery images found')
        return
    if not qry_files:
        print('No query images found')
        return

    print(f'Loaded {len(gal_files)} gallery and {len(qry_files)} query images')
    g_embs = extract_embeddings(model, is_reid, gal_files, device, args.size, args.batch)
    q_embs = extract_embeddings(model, is_reid, qry_files, device, args.size, args.batch)

    # Aggregate gallery to multi-view matching: take max similarity over gallery views
    sims = cosine_sim(q_embs, g_embs)  # shape: Q x G
    max_sims = sims.max(axis=1)
    max_idx = sims.argmax(axis=1)

    # Write CSV
    import csv
    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['query_path', 'best_gallery', 'similarity', 'is_match'])
        for qi, qpath in enumerate(qry_files):
            gi = int(max_idx[qi])
            score = float(max_sims[qi])
            match = 1 if score >= args.threshold else 0
            w.writerow([str(qpath), str(gal_files[gi]), f'{score:.4f}', match])
    print(f'Wrote {args.out}. Threshold {args.threshold} yielded {(max_sims >= args.threshold).sum()} matches.')

if __name__ == '__main__':
    main()
