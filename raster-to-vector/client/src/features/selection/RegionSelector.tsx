import { useState, useRef, useCallback, useEffect } from "react";

export interface CropRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

interface Props {
  imageUrl: string;
  imageWidth: number;
  imageHeight: number;
  onCropChange: (crop: CropRect | null) => void;
}

export function RegionSelector({
  imageUrl,
  imageWidth,
  imageHeight,
  onCropChange,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [drawing, setDrawing] = useState(false);
  const [startPos, setStartPos] = useState<{ x: number; y: number } | null>(
    null,
  );
  const [crop, setCrop] = useState<CropRect | null>(null);
  const [scale, setScale] = useState(1);
  const imgRef = useRef<HTMLImageElement | null>(null);

  // Load image
  useEffect(() => {
    const img = new Image();
    img.src = imageUrl;
    img.onload = () => {
      imgRef.current = img;
      drawCanvas(img, null);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imageUrl]);

  const drawCanvas = useCallback(
    (img: HTMLImageElement, rect: CropRect | null) => {
      const canvas = canvasRef.current;
      const container = containerRef.current;
      if (!canvas || !container) return;

      const maxW = container.clientWidth;
      const s = Math.min(1, maxW / imageWidth);
      setScale(s);

      canvas.width = imageWidth * s;
      canvas.height = imageHeight * s;

      const ctx = canvas.getContext("2d")!;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

      if (rect) {
        // Darken outside
        ctx.fillStyle = "rgba(0,0,0,0.4)";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Clear crop region
        ctx.clearRect(rect.x * s, rect.y * s, rect.width * s, rect.height * s);
        ctx.drawImage(
          img,
          rect.x,
          rect.y,
          rect.width,
          rect.height,
          rect.x * s,
          rect.y * s,
          rect.width * s,
          rect.height * s,
        );

        // Draw crop border
        ctx.strokeStyle = "#3b82f6";
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 3]);
        ctx.strokeRect(rect.x * s, rect.y * s, rect.width * s, rect.height * s);
        ctx.setLineDash([]);
      }
    },
    [imageWidth, imageHeight],
  );

  const getImageCoords = (e: React.MouseEvent) => {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    return {
      x: Math.round((e.clientX - rect.left) / scale),
      y: Math.round((e.clientY - rect.top) / scale),
    };
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    const pos = getImageCoords(e);
    setStartPos(pos);
    setDrawing(true);
    setCrop(null);
    onCropChange(null);
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!drawing || !startPos || !imgRef.current) return;
    const pos = getImageCoords(e);
    const rect: CropRect = {
      x: Math.min(startPos.x, pos.x),
      y: Math.min(startPos.y, pos.y),
      width: Math.abs(pos.x - startPos.x),
      height: Math.abs(pos.y - startPos.y),
    };
    drawCanvas(imgRef.current, rect);
  };

  const handleMouseUp = (e: React.MouseEvent) => {
    if (!drawing || !startPos || !imgRef.current) return;
    setDrawing(false);
    const pos = getImageCoords(e);
    const rect: CropRect = {
      x: Math.max(0, Math.min(startPos.x, pos.x)),
      y: Math.max(0, Math.min(startPos.y, pos.y)),
      width: Math.min(Math.abs(pos.x - startPos.x), imageWidth),
      height: Math.min(Math.abs(pos.y - startPos.y), imageHeight),
    };
    if (rect.width > 5 && rect.height > 5) {
      setCrop(rect);
      onCropChange(rect);
      drawCanvas(imgRef.current, rect);
    } else {
      setCrop(null);
      onCropChange(null);
      drawCanvas(imgRef.current, null);
    }
  };

  const handleClear = () => {
    setCrop(null);
    onCropChange(null);
    if (imgRef.current) drawCanvas(imgRef.current, null);
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <p className="text-sm text-gray-400">
          {crop
            ? `Selected: ${crop.width}×${crop.height} at (${crop.x}, ${crop.y})`
            : "Click and drag to select a region, or vectorize the full image"}
        </p>
        {crop && (
          <button
            onClick={handleClear}
            className="text-xs px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded transition-colors"
          >
            Clear Selection
          </button>
        )}
      </div>
      <div
        ref={containerRef}
        className="w-full overflow-hidden rounded-lg bg-gray-900"
      >
        <canvas
          ref={canvasRef}
          className="cursor-crosshair"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={() => {
            if (drawing) setDrawing(false);
          }}
        />
      </div>
    </div>
  );
}
