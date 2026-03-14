"""API routes for raster-to-vector."""

import uuid
import base64

import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from ..schemas import (
    VectorizeResponse,
    UploadResponse,
    CompareResponse,
    ComparisonMetrics,
)
from ..core.multilevel import multilevel_vectorize, generate_svg
from ..core.comparison import compare

router = APIRouter()

# In-memory storage for uploaded images and results
_images: dict[str, np.ndarray] = {}
_results: dict[str, dict] = {}


@router.post("/upload", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...)):
    """Upload a raster image for vectorization."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    image_id = uuid.uuid4().hex[:12]
    _images[image_id] = image
    h, w = image.shape[:2]

    return UploadResponse(
        image_id=image_id,
        width=w,
        height=h,
        filename=file.filename or "unknown",
    )


@router.post("/vectorize", response_model=VectorizeResponse)
async def vectorize_endpoint(
    image_id: str = Form(...),
    crop_x: int = Form(0),
    crop_y: int = Form(0),
    crop_width: int = Form(0),
    crop_height: int = Form(0),
    remove_background: bool = Form(True),
    num_levels: int = Form(24),
):
    """Run colour-quantization vectorization to produce smooth vector output."""
    if image_id not in _images:
        raise HTTPException(status_code=404, detail="Image not found")

    image = _images[image_id]
    h, w = image.shape[:2]

    # Apply crop if specified
    if crop_width > 0 and crop_height > 0:
        x1 = max(0, crop_x)
        y1 = max(0, crop_y)
        x2 = min(w, crop_x + crop_width)
        y2 = min(h, crop_y + crop_height)
        region = image[y1:y2, x1:x2].copy()
    else:
        region = image.copy()

    # Single pass with shape-accurate defaults
    result = multilevel_vectorize(
        region,
        num_levels=max(2, min(num_levels, 64)),
    )

    svg = generate_svg(result, remove_background=remove_background)
    comparison_svg = generate_svg(result, remove_background=False)
    comp = compare(region, comparison_svg)

    metrics = ComparisonMetrics(
        mae=round(comp.mae, 4),
        ssim=round(comp.ssim_score, 4),
        pixel_diff_ratio=round(comp.pixel_diff_ratio, 4),
    )

    # Store for later comparison
    rid = uuid.uuid4().hex[:12]
    _results[rid] = {
        "svg": svg,
        "image_id": image_id,
        "metrics": metrics,
        "region": region,
    }

    return VectorizeResponse(
        svg=svg,
        width=result.width,
        height=result.height,
        path_count=result.path_count,
        node_count=result.node_count,
        metrics=metrics,
    )


@router.post("/compare", response_model=CompareResponse)
async def compare_images(
    image_id: str = Form(...),
    svg: str = Form(...),
    crop_x: int = Form(0),
    crop_y: int = Form(0),
    crop_width: int = Form(0),
    crop_height: int = Form(0),
):
    """Compare source image with provided SVG."""
    if image_id not in _images:
        raise HTTPException(status_code=404, detail="Image not found")

    image = _images[image_id]
    h, w = image.shape[:2]

    if crop_width > 0 and crop_height > 0:
        x1 = max(0, crop_x)
        y1 = max(0, crop_y)
        x2 = min(w, crop_x + crop_width)
        y2 = min(h, crop_y + crop_height)
        region = image[y1:y2, x1:x2].copy()
    else:
        region = image.copy()

    comp = compare(region, svg)

    # Encode comparison images as base64 data URLs
    overlay_b64 = _encode_image_b64(comp.overlay)
    heatmap_b64 = _encode_image_b64(comp.heatmap)

    return CompareResponse(
        metrics=ComparisonMetrics(
            mae=round(comp.mae, 4),
            ssim=round(comp.ssim_score, 4),
            pixel_diff_ratio=round(comp.pixel_diff_ratio, 4),
        ),
        overlay_url=f"data:image/png;base64,{overlay_b64}",
        heatmap_url=f"data:image/png;base64,{heatmap_b64}",
    )


@router.get("/result/{image_id}")
async def get_result(image_id: str):
    """Get the original image as base64 for display."""
    if image_id not in _images:
        raise HTTPException(status_code=404, detail="Image not found")

    image = _images[image_id]
    b64 = _encode_image_b64(image)

    return {
        "image_id": image_id,
        "image_url": f"data:image/png;base64,{b64}",
        "width": image.shape[1],
        "height": image.shape[0],
    }


def _encode_image_b64(image: np.ndarray) -> str:
    """Encode a numpy image as base64 PNG string."""
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode image")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")
