/**
 * Pyodide-powered in-browser vectorization API.
 * Used on GitHub Pages where no backend server is available.
 */

import type { UploadResponse, VectorizeResponse, VectorizeParams } from "./api";

let worker: Worker | null = null;
let readyResolve: (() => void) | null = null;
let readyReject: ((e: Error) => void) | null = null;
let readyPromise: Promise<void> | null = null;
let progressCallback: ((msg: string) => void) | null = null;
let storedFile: File | null = null;

/** Start loading Pyodide. Call as early as possible. */
export function initPyodide(onProgress?: (msg: string) => void): Promise<void> {
  if (readyPromise) return readyPromise;

  progressCallback = onProgress ?? null;

  readyPromise = new Promise<void>((resolve, reject) => {
    readyResolve = resolve;
    readyReject = reject;
  });

  const base = import.meta.env.BASE_URL ?? "/";
  worker = new Worker(base + "vectorize.worker.js");

  worker.onmessage = (e: MessageEvent) => {
    const msg = e.data;
    if (msg.type === "progress" && progressCallback) {
      progressCallback(msg.message);
    } else if (msg.type === "ready") {
      readyResolve?.();
    } else if (msg.type === "error" && readyResolve) {
      // Error during init
      readyReject?.(new Error(msg.error));
    }
  };

  worker.onerror = (e) => {
    readyReject?.(new Error(String(e.message ?? e)));
  };

  worker.postMessage({ type: "init" });
  return readyPromise;
}

export function isReady(): boolean {
  return readyPromise !== null;
}

/** Store file locally — no server upload needed. */
export async function uploadImageLocal(file: File): Promise<UploadResponse> {
  storedFile = file;
  const img = await createImageBitmap(file);
  const result: UploadResponse = {
    image_id: "local",
    width: img.width,
    height: img.height,
    filename: file.name,
  };
  img.close();
  return result;
}

/** Run vectorization in the Pyodide web worker. */
export async function vectorizeLocal(
  params: VectorizeParams,
): Promise<VectorizeResponse> {
  if (!worker) throw new Error("Pyodide not initialized");
  if (!storedFile) throw new Error("No image uploaded");

  await readyPromise;

  const buffer = await storedFile.arrayBuffer();

  return new Promise<VectorizeResponse>((resolve, reject) => {
    const handler = (e: MessageEvent) => {
      const msg = e.data;
      if (msg.type === "progress" && progressCallback) {
        progressCallback(msg.message);
      } else if (msg.type === "result") {
        worker!.onmessage = null;
        resolve({
          svg: msg.svg,
          width: msg.width,
          height: msg.height,
          path_count: msg.pathCount,
          node_count: msg.nodeCount,
          metrics: null, // No comparison in browser mode
        });
      } else if (msg.type === "error") {
        worker!.onmessage = null;
        reject(new Error(msg.error));
      }
    };
    worker!.onmessage = handler;
    worker!.postMessage(
      {
        type: "vectorize",
        imageData: buffer,
        cropX: params.crop_x ?? 0,
        cropY: params.crop_y ?? 0,
        cropWidth: params.crop_width ?? 0,
        cropHeight: params.crop_height ?? 0,
        removeBackground: params.remove_background ?? true,
        numLevels: params.num_levels ?? 24,
      },
      [buffer],
    );
  });
}
