import { uploadImageLocal, vectorizeLocal } from "./pyodide-api";

export const IS_PAGES = import.meta.env.VITE_GITHUB_PAGES === "true";
const API_BASE = IS_PAGES ? "" : "/api";

export interface UploadResponse {
  image_id: string;
  width: number;
  height: number;
  filename: string;
}

export interface ComparisonMetrics {
  mae: number;
  ssim: number;
  pixel_diff_ratio: number;
}

export interface VectorizeResponse {
  svg: string;
  width: number;
  height: number;
  path_count: number;
  node_count: number;
  metrics: ComparisonMetrics | null;
}

export interface CompareResponse {
  metrics: ComparisonMetrics;
  overlay_url: string;
  heatmap_url: string;
}

export interface ResultResponse {
  image_id: string;
  image_url: string;
  width: number;
  height: number;
}

export async function uploadImage(file: File): Promise<UploadResponse> {
  if (IS_PAGES) return uploadImageLocal(file);

  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/upload`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`Upload failed: ${res.statusText}`);
  return res.json();
}

export interface VectorizeParams {
  image_id: string;
  crop_x?: number;
  crop_y?: number;
  crop_width?: number;
  crop_height?: number;
  remove_background?: boolean;
  num_levels?: number;
}

export async function vectorize(
  params: VectorizeParams,
): Promise<VectorizeResponse> {
  if (IS_PAGES) return vectorizeLocal(params);

  const form = new FormData();
  form.append("image_id", params.image_id);
  form.append("crop_x", String(params.crop_x ?? 0));
  form.append("crop_y", String(params.crop_y ?? 0));
  form.append("crop_width", String(params.crop_width ?? 0));
  form.append("crop_height", String(params.crop_height ?? 0));
  form.append("remove_background", String(params.remove_background ?? true));
  if (params.num_levels != null)
    form.append("num_levels", String(params.num_levels));

  const res = await fetch(`${API_BASE}/vectorize`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error(`Vectorize failed: ${res.statusText}`);
  return res.json();
}

export async function compareImages(
  imageId: string,
  svg: string,
  crop?: { x: number; y: number; width: number; height: number },
): Promise<CompareResponse> {
  const form = new FormData();
  form.append("image_id", imageId);
  form.append("svg", svg);
  form.append("crop_x", String(crop?.x ?? 0));
  form.append("crop_y", String(crop?.y ?? 0));
  form.append("crop_width", String(crop?.width ?? 0));
  form.append("crop_height", String(crop?.height ?? 0));

  const res = await fetch(`${API_BASE}/compare`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error(`Compare failed: ${res.statusText}`);
  return res.json();
}

export async function getResult(imageId: string): Promise<ResultResponse> {
  const res = await fetch(`${API_BASE}/result/${encodeURIComponent(imageId)}`);
  if (!res.ok) throw new Error(`Get result failed: ${res.statusText}`);
  return res.json();
}
