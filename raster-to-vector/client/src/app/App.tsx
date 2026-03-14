import { useState, useCallback } from "react";
import { ImageUpload } from "../features/upload/ImageUpload";
import {
  RegionSelector,
  type CropRect,
} from "../features/selection/RegionSelector";
import { VectorPreview } from "../features/preview/VectorPreview";
import { ComparisonView } from "../features/comparison/ComparisonView";
import { ExportButton } from "../features/export/ExportButton";
import {
  type UploadResponse,
  type VectorizeResponse,
  type CompareResponse,
  vectorize,
  compareImages,
} from "../utils/api";

export function App() {
  // Image state
  const [uploadResult, setUploadResult] = useState<UploadResponse | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [crop, setCrop] = useState<CropRect | null>(null);

  // Vectorize state
  const [vectorResult, setVectorResult] = useState<VectorizeResponse | null>(
    null,
  );
  const [vectorizing, setVectorizing] = useState(false);
  const [vectorError, setVectorError] = useState<string | null>(null);
  const [removeBackground, setRemoveBackground] = useState(true);

  // Comparison state
  const [compareResult, setCompareResult] = useState<CompareResponse | null>(
    null,
  );
  const [comparing, setComparing] = useState(false);

  const handleUpload = useCallback((result: UploadResponse, url: string) => {
    setUploadResult(result);
    setPreviewUrl(url);
    setVectorResult(null);
    setCompareResult(null);
    setCrop(null);
  }, []);

  const handleVectorize = async () => {
    if (!uploadResult) return;
    setVectorizing(true);
    setVectorError(null);
    setCompareResult(null);
    try {
      const result = await vectorize({
        image_id: uploadResult.image_id,
        crop_x: crop?.x,
        crop_y: crop?.y,
        crop_width: crop?.width,
        crop_height: crop?.height,
        remove_background: removeBackground,
      });
      setVectorResult(result);
    } catch (err) {
      setVectorError(
        err instanceof Error ? err.message : "Vectorization failed",
      );
    } finally {
      setVectorizing(false);
    }
  };

  const handleCompare = async () => {
    if (!uploadResult || !vectorResult) return;
    setComparing(true);
    try {
      const result = await compareImages(
        uploadResult.image_id,
        vectorResult.svg,
        crop ?? undefined,
      );
      setCompareResult(result);
    } catch {
      // metrics are shown from vectorization already
    } finally {
      setComparing(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto px-6 py-8 space-y-8">
      {/* Header */}
      <header className="space-y-1">
        <h1 className="text-2xl font-bold tracking-tight">Raster → Vector</h1>
        <p className="text-sm text-gray-500">
          High-quality raster-to-vector conversion with true SVG geometry
        </p>
      </header>

      {/* Upload */}
      {!uploadResult && <ImageUpload onUpload={handleUpload} />}

      {/* Workspace */}
      {uploadResult && previewUrl && (
        <div className="space-y-6">
          {/* Region selection */}
          <RegionSelector
            imageUrl={previewUrl}
            imageWidth={uploadResult.width}
            imageHeight={uploadResult.height}
            onCropChange={setCrop}
          />

          {/* Controls */}
          <div className="flex items-center gap-3 flex-wrap">
            <button
              onClick={handleVectorize}
              disabled={vectorizing}
              className="px-5 py-2 bg-blue-600 hover:bg-blue-700 disabled:opacity-50
                         rounded-lg text-sm font-medium transition-colors"
            >
              {vectorizing
                ? "Vectorizing…"
                : crop
                  ? "Vectorize Selection"
                  : "Vectorize Full Image"}
            </button>

            {/* Background toggle */}
            <label className="flex items-center gap-2 text-sm text-gray-300 cursor-pointer select-none">
              <div
                onClick={() => setRemoveBackground(!removeBackground)}
                className={`relative w-9 h-5 rounded-full transition-colors ${
                  removeBackground ? "bg-blue-600" : "bg-gray-600"
                }`}
              >
                <div
                  className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform ${
                    removeBackground ? "translate-x-4" : ""
                  }`}
                />
              </div>
              Remove Background
            </label>

            {vectorResult && (
              <button
                onClick={handleCompare}
                disabled={comparing}
                className="px-5 py-2 bg-violet-600 hover:bg-violet-700 disabled:opacity-50
                           rounded-lg text-sm font-medium transition-colors"
              >
                {comparing ? "Comparing…" : "Run Comparison"}
              </button>
            )}

            {vectorResult && <ExportButton svgString={vectorResult.svg} />}

            <button
              onClick={() => {
                setUploadResult(null);
                setPreviewUrl(null);
                setVectorResult(null);
                setCompareResult(null);
              }}
              className="px-5 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm transition-colors ml-auto"
            >
              New Image
            </button>
          </div>

          {vectorError && <p className="text-red-400 text-sm">{vectorError}</p>}

          {/* Vector preview */}
          {vectorResult && (
            <div className="space-y-2">
              <div className="flex gap-4 text-xs text-gray-400">
                <span>
                  Paths:{" "}
                  <span className="text-gray-200 font-mono">
                    {vectorResult.path_count}
                  </span>
                </span>
                <span>
                  Nodes:{" "}
                  <span className="text-gray-200 font-mono">
                    {vectorResult.node_count}
                  </span>
                </span>
                <span>
                  Size:{" "}
                  <span className="text-gray-200 font-mono">
                    {vectorResult.width}×{vectorResult.height}
                  </span>
                </span>
              </div>
              <VectorPreview
                svgString={vectorResult.svg}
                width={vectorResult.width}
                height={vectorResult.height}
              />
            </div>
          )}

          {/* Comparison */}
          {vectorResult && vectorResult.metrics && (
            <ComparisonView
              metrics={vectorResult.metrics}
              overlayUrl={compareResult?.overlay_url}
              heatmapUrl={compareResult?.heatmap_url}
              svgString={vectorResult.svg}
              originalUrl={previewUrl}
              width={vectorResult.width}
              height={vectorResult.height}
            />
          )}
        </div>
      )}
    </div>
  );
}
