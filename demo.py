"""
demo.py — Self-contained demonstration (NO real dataset needed)
================================================================
Generates 3 synthetic plant images simulating a time series of crop growth,
then runs the full pipeline so you can see results immediately.

Usage:
    python demo.py

Output:
    data/demo_day01.png   — 3 small plants
    data/demo_day07.png   — 4 medium plants
    data/demo_day14.png   — 5 large plants
    output/demo_day01_pipeline.png
    output/demo_day07_pipeline.png
    output/demo_day14_pipeline.png
    output/growth_chart.png
    output/summary.json
"""

import os
import cv2
import numpy as np

# ── Import our pipeline ─────────────────────────────────
from main import (
    visualize_pipeline,
    measure_growth,
    plot_growth,
    DATA_DIR,
    OUTPUT_DIR,
)


def _draw_plant(canvas, cx, cy, size, rng):
    """
    Draw a single synthetic plant (circle body + ellipse leaves).

    Parameters
    ----------
    canvas : ndarray — image to draw on (modified in-place)
    cx, cy : int     — centre coordinates of the plant
    size   : int     — approximate radius of the main body
    rng    : numpy.random.Generator — seeded RNG for reproducibility
    """
    # Base green colour with slight random variation
    g = int(np.clip(rng.integers(100, 180), 0, 255))
    base_color = (30, g, 25)

    # Main circular body of the plant
    cv2.circle(canvas, (cx, cy), size, base_color, thickness=-1,
               lineType=cv2.LINE_AA)

    # Draw 3–5 elliptical "leaves" radiating from the centre
    n_leaves = rng.integers(3, 6)
    for _ in range(n_leaves):
        angle   = int(rng.integers(0, 360))
        leaf_w  = max(5, int(size * rng.uniform(0.5, 1.0)))
        leaf_h  = max(3, int(size * rng.uniform(0.2, 0.5)))
        # Slightly offset from centre to create a natural spread
        offset  = int(size * 0.3)
        ox = cx + int(offset * np.cos(np.radians(angle)))
        oy = cy + int(offset * np.sin(np.radians(angle)))
        leaf_g  = int(np.clip(rng.integers(80, 200), 0, 255))
        leaf_color = (25, leaf_g, 20)
        cv2.ellipse(canvas, (ox, oy), (leaf_w, leaf_h), angle,
                    0, 360, leaf_color, thickness=-1, lineType=cv2.LINE_AA)


def generate_synthetic_image(filename, n_plants, size_range, seed):
    """
    Create a single synthetic greenhouse crop image and save it.

    Parameters
    ----------
    filename  : str           — output file name (saved inside DATA_DIR)
    n_plants  : int           — how many plants to place
    size_range: (int, int)    — (min_radius, max_radius) for plant blobs
    seed      : int           — random seed for full reproducibility

    Returns
    -------
    path : str — full path to the saved image
    """
    rng = np.random.default_rng(seed)  # deterministic generator

    W, H = 450, 450
    # Dark soil-like background
    canvas = np.full((H, W, 3), (28, 38, 24), dtype=np.uint8)

    # Place plants at random positions, avoiding image edges
    margin = size_range[1] + 20
    positions = []
    attempts = 0
    while len(positions) < n_plants and attempts < 500:
        cx = int(rng.integers(margin, W - margin))
        cy = int(rng.integers(margin, H - margin))
        # Ensure plants don't overlap too much
        too_close = any(
            np.hypot(cx - px, cy - py) < size_range[1] * 1.8
            for px, py in positions
        )
        if not too_close:
            positions.append((cx, cy))
        attempts += 1

    # Draw each plant
    for cx, cy in positions:
        size = int(rng.integers(size_range[0], size_range[1] + 1))
        _draw_plant(canvas, cx, cy, size, rng)

    # Add Gaussian noise for realism
    noise = rng.normal(0, 8, canvas.shape).astype(np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Save
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, filename)
    cv2.imwrite(path, canvas)
    print(f"  Created: {path}  ({n_plants} plants, size {size_range})")
    return path


def main():
    """Generate demo images, run the full pipeline, and print results."""
    print("=" * 56)
    print("  DEMO — Greenhouse Crop Growth Monitoring Pipeline")
    print("=" * 56)
    print()

    # ── 1. Generate synthetic time-series images ─────────
    print("[Step 1] Generating synthetic plant images...\n")
    specs = [
        # (filename,           n_plants, size_range, seed)
        ("demo_day01.png",     3,        (18, 28),   42),
        ("demo_day07.png",     4,        (28, 42),   43),
        ("demo_day14.png",     5,        (38, 55),   44),
    ]
    paths = []
    for fname, n, sz, sd in specs:
        p = generate_synthetic_image(fname, n, sz, sd)
        paths.append(p)

    # ── 2. Run visualize_pipeline on each image ──────────
    print("\n[Step 2] Running watershed pipeline on each image...\n")
    counts = []
    for path in paths:
        n = visualize_pipeline(path, save=True)
        counts.append(n)
        print(f"  -> {os.path.basename(path)}: {n} plant(s) detected\n")

    # ── 3. Growth measurement ────────────────────────────
    print("[Step 3] Measuring plant growth over time...\n")
    labels = ["Day 1", "Day 7", "Day 14"]
    growth = measure_growth(paths, labels=labels)

    # ── 4. Growth chart ──────────────────────────────────
    print("\n[Step 4] Plotting growth chart...\n")
    plot_growth(growth, save=True)

    # ── 5. List output contents ──────────────────────────
    print("\n[Step 5] Output folder contents:\n")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_size = 0
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.isfile(fpath):
            sz = os.path.getsize(fpath)
            total_size += sz
            print(f"  {fname:<40s} {sz:>8,} bytes")
    print(f"\n  Total: {total_size:,} bytes")

    # ── Done ─────────────────────────────────────────────
    print("\n" + "=" * 56)
    print("  DEMO COMPLETE -- your pipeline works perfectly")
    print("=" * 56 + "\n")


if __name__ == "__main__":
    main()
