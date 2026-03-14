import { useState } from "react";
import { type ComparisonMetrics } from "../../utils/api";

interface Props {
  metrics: ComparisonMetrics;
  overlayUrl?: string;
  heatmapUrl?: string;
  svgString: string;
  originalUrl: string;
  width: number;
  height: number;
}

type ViewMode = "side-by-side" | "overlay" | "heatmap";

export function ComparisonView({
  metrics,
  overlayUrl,
  heatmapUrl,
  svgString,
  originalUrl,
  width,
  height,
}: Props) {
  const [mode, setMode] = useState<ViewMode>("side-by-side");
  const svgDataUrl = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svgString)}`;

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-4">
        <h3 className="text-sm font-medium text-gray-300">Comparison</h3>
        <div className="flex gap-1 bg-gray-800 rounded-lg p-0.5">
          {(["side-by-side", "overlay", "heatmap"] as ViewMode[]).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`px-3 py-1 text-xs rounded-md transition-colors ${
                mode === m
                  ? "bg-blue-600 text-white"
                  : "text-gray-400 hover:text-gray-200"
              }`}
            >
              {m === "side-by-side"
                ? "Side by Side"
                : m === "overlay"
                  ? "Overlay"
                  : "Heatmap"}
            </button>
          ))}
        </div>
      </div>

      {/* Metrics bar */}
      <div className="flex gap-6 text-xs text-gray-400">
        <span>
          MAE:{" "}
          <span className="text-gray-200 font-mono">
            {metrics.mae.toFixed(2)}
          </span>
        </span>
        <span>
          SSIM:{" "}
          <span className="text-gray-200 font-mono">
            {metrics.ssim.toFixed(4)}
          </span>
        </span>
        <span>
          Diff pixels:{" "}
          <span className="text-gray-200 font-mono">
            {(metrics.pixel_diff_ratio * 100).toFixed(1)}%
          </span>
        </span>
      </div>

      {/* Views */}
      <div className="bg-gray-900 rounded-lg p-4 overflow-auto">
        {mode === "side-by-side" && (
          <div className="flex gap-4 items-start">
            <div className="space-y-1">
              <span className="text-xs text-gray-500">Original</span>
              <div className="bg-white rounded overflow-hidden">
                <img
                  src={originalUrl}
                  alt="Original"
                  style={{ maxWidth: width, maxHeight: height }}
                />
              </div>
            </div>
            <div className="space-y-1">
              <span className="text-xs text-gray-500">Vector</span>
              <div className="bg-white rounded overflow-hidden">
                <img
                  src={svgDataUrl}
                  alt="Vector"
                  style={{ maxWidth: width, maxHeight: height }}
                />
              </div>
            </div>
          </div>
        )}
        {mode === "overlay" && overlayUrl && (
          <div className="space-y-1">
            <span className="text-xs text-gray-500">Overlay (50/50 blend)</span>
            <img
              src={overlayUrl}
              alt="Overlay"
              className="rounded"
              style={{ maxWidth: width }}
            />
          </div>
        )}
        {mode === "heatmap" && heatmapUrl && (
          <div className="space-y-1">
            <span className="text-xs text-gray-500">Difference Heatmap</span>
            <img
              src={heatmapUrl}
              alt="Heatmap"
              className="rounded"
              style={{ maxWidth: width }}
            />
          </div>
        )}
        {mode === "overlay" && !overlayUrl && (
          <p className="text-gray-500 text-sm">Run comparison to see overlay</p>
        )}
        {mode === "heatmap" && !heatmapUrl && (
          <p className="text-gray-500 text-sm">Run comparison to see heatmap</p>
        )}
      </div>
    </div>
  );
}
