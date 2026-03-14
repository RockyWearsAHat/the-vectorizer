import { useCallback } from "react";
import { uploadImage, type UploadResponse } from "../../utils/api";
import { useAsync } from "../../hooks/useAsync";

interface Props {
  onUpload: (result: UploadResponse, previewUrl: string) => void;
}

export function ImageUpload({ onUpload }: Props) {
  const { loading, error, execute } = useAsync(uploadImage);

  const handleDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file) await processFile(file);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  const handleSelect = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) await processFile(file);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  async function processFile(file: File) {
    const result = await execute(file);
    if (result) {
      const previewUrl = URL.createObjectURL(file);
      onUpload(result, previewUrl);
    }
  }

  return (
    <div
      onDrop={handleDrop}
      onDragOver={(e) => e.preventDefault()}
      className="border-2 border-dashed border-gray-600 rounded-xl p-10
                 flex flex-col items-center justify-center gap-4
                 hover:border-blue-500 transition-colors cursor-pointer"
    >
      <svg
        className="w-12 h-12 text-gray-500"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={1.5}
          d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
        />
      </svg>
      <p className="text-gray-400">Drop an image here or click to select</p>
      <label className="px-5 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-sm font-medium cursor-pointer transition-colors">
        Choose File
        <input
          type="file"
          accept="image/*"
          className="hidden"
          onChange={handleSelect}
        />
      </label>
      {loading && <p className="text-blue-400 text-sm">Uploading…</p>}
      {error && <p className="text-red-400 text-sm">{error}</p>}
    </div>
  );
}
