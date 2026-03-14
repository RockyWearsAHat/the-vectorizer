"""Sweep denoising strategies: find the best color-gated approach.

Goal: Remove SD per-pixel static within flat regions while
preserving edges between different colors.
"""
import cv2
import numpy as np
import time
from app.core.multilevel import (
    multilevel_vectorize, generate_svg,
    _merge_close_clusters, _compute_edge_weight, detect_background,
    _bgr_to_hex, _polygon_area, _fit_contour, VectorLayer, MultilevelResult,
)
from app.core.comparison import compare
from skimage.measure import find_contours


def run_pipeline(img, denoised, label=""):
    """Run the full vectorization pipeline with a custom denoised image."""
    h, w = img.shape[:2]
    bg_color, _ = detect_background(img)
    bg_hex = _bgr_to_hex(bg_color)
    edge_weight = _compute_edge_weight(img)

    pixels = denoised.reshape(-1, 3).astype(np.float32)
    K = 24
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels, K, None, criteria, 6, cv2.KMEANS_PP_CENTERS
    )
    centers, labels = _merge_close_clusters(
        centers, labels.flatten(), h, w, threshold=60.0,
    )
    num_clusters = len(centers)
    centers_u = centers.astype(np.uint8)
    centers_f = centers.astype(np.float32)

    bg_dists = np.array([
        np.linalg.norm(centers_f[k] - bg_color.astype(np.float32))
        for k in range(num_clusters)
    ])
    bg_cluster_idx = int(np.argmin(bg_dists))
    bg_cluster = bg_cluster_idx if bg_dists[bg_cluster_idx] < 40.0 else -1

    # Distance map from ORIGINAL image
    pixels_3d = img.astype(np.float32)
    dist_map = np.empty((h, w, num_clusters), dtype=np.float32)
    for k in range(num_clusters):
        diff = pixels_3d - centers_f[k]
        dist_map[:, :, k] = np.sqrt(np.sum(diff * diff, axis=2))

    grays = np.array([
        int(cv2.cvtColor(c.reshape(1, 1, 3), cv2.COLOR_BGR2GRAY)[0, 0])
        for c in centers_u
    ])
    order = np.argsort(-grays)

    layers = []
    for cluster_idx in order:
        if cluster_idx == bg_cluster:
            continue
        color_hex = _bgr_to_hex(centers_u[cluster_idx])
        d_k = dist_map[:, :, cluster_idx]
        other_mask = np.ones(num_clusters, dtype=bool)
        other_mask[cluster_idx] = False
        d_other = np.min(dist_map[:, :, other_mask], axis=2)
        denom = d_k + d_other
        denom = np.where(denom < 1e-10, 1e-10, denom)
        soft_raw = d_other / denom

        soft_crisp = cv2.GaussianBlur(soft_raw, (0, 0), sigmaX=0.6)
        soft_smooth = cv2.GaussianBlur(soft_raw, (0, 0), sigmaX=1.5)
        soft = edge_weight * soft_crisp + (1.0 - edge_weight) * soft_smooth

        iso_levels = [0.20, 0.50]
        iso_opacities = [0.50, 1.00]

        layer_paths = []
        layer_opacities = []
        for iso, opacity in zip(iso_levels, iso_opacities):
            contour_list = find_contours(soft, iso)
            iso_parts = []
            for contour in contour_list:
                if len(contour) < 4:
                    continue
                xy = contour[:, ::-1].astype(np.float64)
                area = abs(_polygon_area(xy))
                if area < 15:
                    continue
                d = _fit_contour(xy, 0.15, 0.2, 60.0)
                if d:
                    iso_parts.append(d)
            if iso_parts:
                combined = " ".join(iso_parts)
                layer_paths.append(combined)
                layer_opacities.append(opacity)

        if layer_paths:
            layers.append(VectorLayer(
                paths=layer_paths, opacities=layer_opacities, color=color_hex,
            ))

    mr = MultilevelResult(
        layers=layers, width=w, height=h,
        background_color=bg_hex, path_count=0, node_count=0,
    )
    svg = generate_svg(mr, remove_background=False)
    comp = compare(img, svg)
    return comp.ssim_score, comp.mae, len(layers), svg


# Load images
ref = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")
h, w = ref.shape[:2]
crop = ref[0:min(410, h), max(0, (w - 564) // 2):max(0, (w - 564) // 2) + 564]
mahal = cv2.imread("/tmp/mahal_right.png")

images = {"crop": crop, "mahal": mahal}

# --- Define denoising strategies ---
def denoise_bilateral_current(img):
    """Current: bilateral d=7, sc=10, ss=10"""
    return cv2.bilateralFilter(img, 7, 10, 10)

def denoise_bilateral_wide_tight(img):
    """Wide spatial, tight color: d=15, sc=12, ss=30"""
    return cv2.bilateralFilter(img, 15, 12, 30)

def denoise_bilateral_wide_medium(img):
    """Wide spatial, medium color: d=15, sc=20, ss=30"""
    return cv2.bilateralFilter(img, 15, 20, 30)

def denoise_bilateral_vwide_tight(img):
    """Very wide spatial, tight color: d=21, sc=15, ss=40"""
    return cv2.bilateralFilter(img, 21, 15, 40)

def denoise_nlm(img):
    """Non-local means: h=6, hColor=6, template=7, search=21"""
    return cv2.fastNlMeansDenoisingColored(img, None, 6, 6, 7, 21)

def denoise_nlm_strong(img):
    """Non-local means stronger: h=10, hColor=10"""
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

def denoise_bilateral_then_none(img):
    """No denoising at all - raw image"""
    return img.copy()

def denoise_median(img):
    """Median filter 3x3 - good at salt-and-pepper"""
    return cv2.medianBlur(img, 3)

def denoise_bilateral_huge_tight(img):
    """Huge spatial, very tight color: d=25, sc=10, ss=50"""
    return cv2.bilateralFilter(img, 25, 10, 50)

def denoise_bilateral_adaptive(img):
    """Two-pass: first tight bilateral, then edge-gated second pass.
    Run a second bilateral only on flat areas."""
    first = cv2.bilateralFilter(img, 7, 10, 10)
    edge_w = _compute_edge_weight(img)
    second = cv2.bilateralFilter(first, 15, 15, 30)
    # Blend: edges keep first pass, flat areas get second pass
    ew3 = edge_w[:, :, np.newaxis]
    result = (ew3 * first.astype(np.float32) + (1.0 - ew3) * second.astype(np.float32))
    return np.clip(result, 0, 255).astype(np.uint8)


strategies = [
    ("current (bi 7/10/10)", denoise_bilateral_current),
    ("no denoise (raw)", denoise_bilateral_then_none),
    ("median 3x3", denoise_median),
    ("bi wide tight 15/12/30", denoise_bilateral_wide_tight),
    ("bi wide med 15/20/30", denoise_bilateral_wide_medium),
    ("bi vwide tight 21/15/40", denoise_bilateral_vwide_tight),
    ("bi huge tight 25/10/50", denoise_bilateral_huge_tight),
    ("NLM 6/6", denoise_nlm),
    ("NLM 10/10", denoise_nlm_strong),
    ("adaptive 2-pass", denoise_bilateral_adaptive),
]

print(f"{'Strategy':>28s}  {'crop SSIM':>10s}  {'mahal SSIM':>10s}  {'avg':>6s}  {'time':>5s}")
print("-" * 70)

for strat_name, strat_fn in strategies:
    ssims = {}
    t0 = time.time()
    for name, img in images.items():
        if img is None:
            continue
        denoised = strat_fn(img)
        ssim, mae, nlayers, svg = run_pipeline(img, denoised, label=strat_name)
        ssims[name] = ssim
        # Save best SVGs for visual inspection
        if name == "crop":
            with open(f"/tmp/denoise_{strat_name.replace(' ', '_').replace('/', '_')}_crop.svg", "w") as f:
                f.write(svg)
    dt = time.time() - t0
    avg = np.mean(list(ssims.values()))
    print(f"  {strat_name:>26s}  {ssims.get('crop', 0):.4f}      {ssims.get('mahal', 0):.4f}   {avg:.4f}  {dt:.1f}s")
