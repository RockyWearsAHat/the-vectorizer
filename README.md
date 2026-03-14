# SVG-gen — High-Quality Raster-to-Vector Conversion

Convert raster images (PNG, JPG, WebP) to clean SVG vector graphics with
sub-pixel accuracy. Black ink and line art on white paper works especially well.

## Quick Start

### Prerequisites

- **Python 3.12+** with pip
- **Node.js 18+** with npm
- **Cairo** graphics library (for SVG rasterization/comparison)
  - macOS: `brew install cairo`
  - Ubuntu: `sudo apt install libcairo2-dev`

### Run Locally

```bash
# 1. Clone
git clone https://github.com/alexwaldmann/SVG-gen.git
cd SVG-gen

# 2. Backend
cd raster-to-vector/server
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# macOS with Homebrew Cairo:
DYLD_LIBRARY_PATH="/opt/homebrew/opt/cairo/lib" \
  uvicorn app.main:app --reload --host 127.0.0.1 --port 8100

# Linux:
uvicorn app.main:app --reload --host 127.0.0.1 --port 8100

# 3. Frontend (separate terminal)
cd raster-to-vector/client
npm install
npm run dev
```

Open **http://localhost:5173** and upload an image.

### GitHub Pages Demo

Visit the [live demo](https://alexwaldmann.github.io/SVG-gen/) — you'll
need to run the backend locally (step 2 above) for it to work. The frontend
connects to `http://127.0.0.1:8100`.

## How It Works

1. **K-means clustering** finds the real colors in the image
2. **Mediator absorption** merges anti-aliased boundary pixels into their
   parent color — no gray splotch artifacts
3. **Soft membership fields** compute how strongly each pixel belongs to
   each color, producing sub-pixel edge locations
4. **6× superresolution** upscaling with bicubic interpolation for smooth
   contours
5. **Adaptive iso-threshold** places the contour at the perceptual edge
6. **Bézier curve fitting** with corner detection produces clean SVG paths

## Project Structure

```
raster-to-vector/
  server/          Python FastAPI backend (port 8100)
    app/
      core/
        multilevel/        Vectorization engine
        curve_fitting/     Bézier fitting
        comparison/        SSIM / MAE metrics
  client/          Vite + React + TypeScript frontend (port 5173)
  shared/          Dev scripts and sample images
```

## License

See [LICENSE](LICENSE). Free for personal and non-commercial use. Output
(SVGs) is yours to use however you want. Redistribution of the software
requires written permission.
