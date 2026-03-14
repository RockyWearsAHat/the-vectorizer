"""API schemas for request/response validation."""

from pydantic import BaseModel


class CropRegion(BaseModel):
    x: int
    y: int
    width: int
    height: int


class VectorizeRequest(BaseModel):
    crop: CropRegion | None = None
    remove_background: bool = True
    num_levels: int = 24


class ComparisonMetrics(BaseModel):
    mae: float
    ssim: float
    pixel_diff_ratio: float


class VectorizeResponse(BaseModel):
    svg: str
    width: int
    height: int
    path_count: int
    node_count: int
    metrics: ComparisonMetrics | None = None


class UploadResponse(BaseModel):
    image_id: str
    width: int
    height: int
    filename: str


class CompareResponse(BaseModel):
    metrics: ComparisonMetrics
    overlay_url: str
    heatmap_url: str
