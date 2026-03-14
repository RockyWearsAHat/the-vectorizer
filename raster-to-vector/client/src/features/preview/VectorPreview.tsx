interface Props {
  svgString: string;
  width: number;
  height: number;
}

export function VectorPreview({ svgString, width, height }: Props) {
  const dataUrl = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svgString)}`;

  return (
    <div className="space-y-2">
      <h3 className="text-sm font-medium text-gray-300">Vector Preview</h3>
      <div className="bg-white rounded-lg overflow-hidden inline-block">
        <img
          src={dataUrl}
          alt="Vector preview"
          style={{ maxWidth: "100%", width, height }}
        />
      </div>
    </div>
  );
}
