/* Pyodide vectorization web worker.
   Loaded as a classic worker from public/ — not bundled by Vite. */

/* global importScripts, loadPyodide, self */

var PYODIDE_CDN = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/";
var pyodide = null;

/** Derive base URL from the worker script location. */
function getBaseUrl() {
  var href = self.location.href;
  return href.substring(0, href.lastIndexOf("/") + 1);
}

async function fetchText(url) {
  var res = await fetch(url);
  if (!res.ok) throw new Error("Failed to fetch " + url + ": " + res.status);
  return await res.text();
}

self.onmessage = async function (e) {
  var msg = e.data;

  if (msg.type === "init") {
    try {
      self.postMessage({ type: "progress", message: "Loading Python runtime…" });
      importScripts(PYODIDE_CDN + "pyodide.js");

      pyodide = await loadPyodide({ indexURL: PYODIDE_CDN });

      self.postMessage({ type: "progress", message: "Installing numpy…" });
      await pyodide.loadPackage("numpy");

      self.postMessage({ type: "progress", message: "Installing scipy…" });
      await pyodide.loadPackage("scipy");

      self.postMessage({ type: "progress", message: "Installing scikit-image…" });
      await pyodide.loadPackage("scikit-image");

      self.postMessage({ type: "progress", message: "Installing OpenCV…" });
      await pyodide.loadPackage("opencv-python");

      self.postMessage({
        type: "progress",
        message: "Loading vectorization engine…",
      });

      var base = getBaseUrl();
      var files = [
        "curve_fitting.py",
        "stroke_reconstruction.py",
        "multilevel.py",
        "vectorize_entry.py",
      ];
      for (var i = 0; i < files.length; i++) {
        var src = await fetchText(base + "python/" + files[i]);
        pyodide.FS.writeFile("/home/pyodide/" + files[i], src);
      }

      await pyodide.runPythonAsync(
        "import sys; sys.path.insert(0, '/home/pyodide')"
      );
      await pyodide.runPythonAsync("import vectorize_entry");

      self.postMessage({ type: "ready" });
    } catch (err) {
      self.postMessage({ type: "error", error: String(err.message || err) });
    }
  }

  if (msg.type === "vectorize") {
    if (!pyodide) {
      self.postMessage({ type: "error", error: "Pyodide not initialized" });
      return;
    }
    try {
      self.postMessage({ type: "progress", message: "Vectorizing…" });

      // Pass image bytes to Python
      var imageArray = new Uint8Array(msg.imageData);
      pyodide.globals.set("_img_bytes", pyodide.toPy(imageArray));
      pyodide.globals.set("_crop_x", msg.cropX || 0);
      pyodide.globals.set("_crop_y", msg.cropY || 0);
      pyodide.globals.set("_crop_w", msg.cropWidth || 0);
      pyodide.globals.set("_crop_h", msg.cropHeight || 0);
      pyodide.globals.set("_rm_bg", msg.removeBackground !== false);
      pyodide.globals.set("_levels", msg.numLevels || 24);

      var resultJson = await pyodide.runPythonAsync(
        "vectorize_entry.run(\n" +
          "    bytes(_img_bytes),\n" +
          "    crop_x=int(_crop_x), crop_y=int(_crop_y),\n" +
          "    crop_w=int(_crop_w), crop_h=int(_crop_h),\n" +
          "    remove_bg=bool(_rm_bg), num_levels=int(_levels),\n" +
          ")"
      );

      var result = JSON.parse(resultJson);
      self.postMessage({
        type: "result",
        svg: result.svg,
        width: result.width,
        height: result.height,
        pathCount: result.path_count,
        nodeCount: result.node_count,
      });
    } catch (err) {
      self.postMessage({ type: "error", error: String(err.message || err) });
    }
  }
};
