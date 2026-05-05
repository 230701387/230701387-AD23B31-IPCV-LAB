"""
main.py — Greenhouse Crop Growth Monitoring using cv2.watershed
================================================================
Core algorithm for per-plant segmentation of greenhouse crop images.

Student  : Roll No 2116230701387
Subject  : Computer Vision & Image Processing
Topic    : 32 — Greenhouse Crop Growth Monitoring
Algorithm: cv2.watershed Per-Plant Segmentation
Dataset  : CVPPP Leaf Segmentation Challenge (A1 subset)

Usage:
    python main.py          → batch-processes all images in data/
    from main import *      → import functions into other scripts
"""

import os
import sys
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — works without display
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ═══════════════════════════════════════════════════════
# TUNABLE CONSTANTS — change these to adjust behaviour
# ═══════════════════════════════════════════════════════
DATA_DIR    = "data"       # folder containing input images
OUTPUT_DIR  = "output"     # folder for all saved results
DIST_THRESH = 0.5          # fraction of max distance-transform value
                           # used to identify sure foreground (0.3–0.7)
MIN_AREA    = 200          # minimum contour area (px²) to count as a plant


# ═══════════════════════════════════════════════════════
# FUNCTION 1 — segment_plants
# ═══════════════════════════════════════════════════════

def segment_plants(img_path, dist_thresh=DIST_THRESH, min_area=MIN_AREA):
    """
    Run the full watershed segmentation pipeline on a single image.

    Parameters
    ----------
    img_path    : str   — path to the input BGR image
    dist_thresh : float — fraction of max distance-transform for sure-fg
    min_area    : int   — minimum contour area (px²) to count as a plant

    Returns
    -------
    result_rgb     : ndarray — RGB image with watershed boundaries drawn
    n_plants       : int     — number of detected plants (area >= min_area)
    markers        : ndarray — watershed marker map (int32)
    dist_transform : ndarray — distance-transform of the binary mask
    thresh         : ndarray — Otsu binary threshold result
    opening        : ndarray — morphologically cleaned binary mask
    """

    # ── Step 1: Load the image ───────────────────────────
    # cv2.imread returns None when the file is missing or corrupt
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(
            f"Cannot read image: '{img_path}'. "
            "Check the path and file format."
        )

    # ── Step 2: Convert BGR → Grayscale ──────────────────
    # Grayscale simplifies thresholding — we only need intensity
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── Step 3: Otsu thresholding ────────────────────────
    # THRESH_BINARY_INV makes foreground (plants) white
    # THRESH_OTSU automatically picks the optimal threshold value
    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # ── Step 4: Morphological opening ────────────────────
    # Removes small noise specks while preserving plant shapes
    # Ellipse kernel is gentler on round plant blobs than a rectangle
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, kernel, iterations=2
    )

    # ── Step 5: Sure background via dilation ─────────────
    # Dilating the mask expands it outward; anything outside is
    # definitely background — gives watershed room to work
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # ── Step 6: Distance transform ───────────────────────
    # Computes Euclidean distance from each white pixel to the nearest
    # black pixel — peaks correspond to plant centres
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    # ── Step 7: Sure foreground via distance threshold ───
    # Pixels with distance > (dist_thresh × max) are almost certainly
    # at the core of a plant — these are our seed regions
    _, sure_fg = cv2.threshold(
        dist_transform, dist_thresh * dist_transform.max(), 255, 0
    )
    sure_fg = np.uint8(sure_fg)  # convert for subtraction below

    # ── Step 8: Unknown region ───────────────────────────
    # The gap between sure_bg and sure_fg is "unknown" — watershed
    # will decide whether these pixels belong to foreground or background
    unknown = cv2.subtract(sure_bg, sure_fg)

    # ── Step 9: Connected-component labelling ────────────
    # Each disconnected blob in sure_fg gets a unique integer label
    n_labels, markers = cv2.connectedComponents(sure_fg)
    # Shift labels by +1 so that background becomes 1 (not 0)
    # because watershed treats 0 as "unknown / to-be-determined"
    markers = markers + 1
    # Mark the unknown region as 0 so watershed will classify it
    markers[unknown == 255] = 0

    # ── Step 10: Watershed ───────────────────────────────
    # The algorithm "floods" from each marker and builds barriers
    # where different flood regions would meet — these are boundaries
    img_copy = img.copy()  # watershed modifies the image in-place
    markers = cv2.watershed(img_copy, markers)

    # ── Step 11: Draw boundaries on the result ───────────
    # Watershed marks boundaries with -1 in the marker map
    # We draw them as red-ish lines on an RGB copy for display
    result_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result_rgb[markers == -1] = [220, 50, 50]

    # ── Step 12: Count valid plant regions ───────────────
    # Each unique marker > 1 is a candidate region (1 = background)
    # We keep only those whose contour area exceeds min_area
    unique_labels = set(np.unique(markers))
    unique_labels.discard(-1)  # boundary marker
    unique_labels.discard(1)   # background marker

    n_plants = 0
    for label in unique_labels:
        # Create a binary mask for this single region
        region_mask = np.uint8(markers == label) * 255
        contours, _ = cv2.findContours(
            region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # Sum the areas of all contours belonging to this label
        area = sum(cv2.contourArea(c) for c in contours)
        if area >= min_area:
            n_plants += 1

    return result_rgb, n_plants, markers, dist_transform, thresh, opening


# ═══════════════════════════════════════════════════════
# FUNCTION 2 — visualize_pipeline
# ═══════════════════════════════════════════════════════

def visualize_pipeline(img_path, save=True):
    """
    Create a 2×2 panel figure showing each pipeline stage.

    Parameters
    ----------
    img_path : str  — path to the input image
    save     : bool — if True, save to output/{name}_pipeline.png

    Returns
    -------
    n_plants : int — number of plants detected
    """
    # Run the full segmentation pipeline
    result_rgb, n_plants, markers, dist, thresh, opening = \
        segment_plants(img_path)

    # Load the original image and convert to RGB for display
    original = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    filename = os.path.basename(img_path)

    # ── Build the 2×2 figure ─────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    fig.patch.set_facecolor("#0f1117")      # dark background
    fig.suptitle(
        f"Greenhouse Crop Monitoring  |  {filename}",
        color="white", fontsize=14, fontweight="bold", y=0.97
    )

    panels = [
        (axes[0, 0], original, "1. Original Image",              None),
        (axes[0, 1], thresh,   "2. Otsu Threshold \u2014 plants=white", "gray"),
        (axes[1, 0], dist,     "3. Distance Transform \u2014 bright=centre", "jet"),
        (axes[1, 1], result_rgb,
         f"4. Result \u2014 {n_plants} plant(s) detected",        None),
    ]

    for ax, data, title, cmap in panels:
        ax.set_facecolor("#0f1117")
        ax.imshow(data, cmap=cmap)
        ax.set_title(title, color="white", fontsize=11, pad=8)
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        stem = os.path.splitext(filename)[0]
        out_path = os.path.join(OUTPUT_DIR, f"{stem}_pipeline.png")
        try:
            fig.savefig(
                out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor()
            )
            print(f"  Saved: {out_path}")
        except OSError as e:
            print(f"  ERROR saving {out_path}: {e}")

    plt.close(fig)  # free memory
    return n_plants


# ═══════════════════════════════════════════════════════
# FUNCTION 3 — measure_growth
# ═══════════════════════════════════════════════════════

def measure_growth(image_paths, labels=None):
    """
    Measure total green plant area in each image.

    Parameters
    ----------
    image_paths : list[str]       — paths to images in chronological order
    labels      : list[str]|None  — human-readable labels for each image

    Returns
    -------
    growth_data : list[(str, int)] — list of (label, total_area_px)
    """
    if labels is None:
        labels = [os.path.basename(p) for p in image_paths]

    growth_data = []
    for path, label in zip(image_paths, labels):
        try:
            img = cv2.imread(path)
            if img is None:
                print(f"  WARNING: cannot read '{path}', skipping.")
                continue
            # Use HSV colour space to isolate green plant pixels
            # This is more robust than grayscale Otsu for dark-soil images
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # Green hue range in HSV (roughly 25-95 on OpenCV 0-179 scale)
            lower_green = np.array([25, 30, 30])
            upper_green = np.array([95, 255, 255])
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            # Clean up with morphological opening
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            clean = cv2.morphologyEx(
                green_mask, cv2.MORPH_OPEN, kernel, iterations=2
            )
            # Find contours and sum their areas
            contours, _ = cv2.findContours(
                clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            total_area = int(sum(cv2.contourArea(c) for c in contours))
            growth_data.append((label, total_area))
            print(f"  {label}: total plant area = {total_area:,} px^2")
        except Exception as e:
            print(f"  ERROR processing '{path}': {e}")

    return growth_data


# ═══════════════════════════════════════════════════════
# FUNCTION 4 — plot_growth
# ═══════════════════════════════════════════════════════

def plot_growth(growth_data, save=True):
    """
    Plot a growth chart from measure_growth output.

    Parameters
    ----------
    growth_data : list[(str, int)] — from measure_growth()
    save        : bool             — save to output/growth_chart.png

    Returns
    -------
    None
    """
    if not growth_data:
        print("  No growth data to plot.")
        return

    labels = [g[0] for g in growth_data]
    areas  = [g[1] for g in growth_data]
    xs     = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#161922")

    # Main line with circle markers
    ax.plot(
        xs, areas,
        color="#4ade80", linewidth=2.5,
        marker="o", markersize=9,
        markerfacecolor="white", markeredgecolor="#4ade80",
        markeredgewidth=2, zorder=5
    )
    # Filled area under the curve for visual depth
    ax.fill_between(xs, areas, alpha=0.12, color="#4ade80")

    # Axis formatting
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, color="white", fontsize=10)
    ax.set_ylabel("Total Plant Area (px^2)", color="white", fontsize=11)
    ax.set_title(
        "Crop Growth Over Time", color="white", fontsize=14, fontweight="bold"
    )

    # Comma-formatted y-axis (e.g. 12,345)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"
    ))
    ax.tick_params(axis="y", colors="white")
    ax.tick_params(axis="x", colors="white")

    # Subtle grid
    ax.grid(True, color="#2a2d35", linewidth=0.8)
    ax.set_axisbelow(True)

    # Spine styling
    for spine in ax.spines.values():
        spine.set_color("#2a2d35")

    plt.tight_layout()

    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, "growth_chart.png")
        try:
            fig.savefig(
                out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor()
            )
            print(f"  Saved: {out_path}")
        except OSError as e:
            print(f"  ERROR saving {out_path}: {e}")

    plt.close(fig)


# ═══════════════════════════════════════════════════════
# FUNCTION 5 — check_accuracy
# ═══════════════════════════════════════════════════════

def check_accuracy(img_path, ground_truth_count):
    """
    Compare detected plant count against a known ground truth.

    Parameters
    ----------
    img_path           : str — path to the image
    ground_truth_count : int — known number of plants

    Returns
    -------
    (detected, accuracy_pct) : (int, float)
    """
    _, detected, *_ = segment_plants(img_path)

    # Accuracy = 100% − percentage error, clamped to [0, 100]
    if ground_truth_count > 0:
        error_pct = abs(detected - ground_truth_count) / ground_truth_count * 100
        accuracy  = max(0.0, 100.0 - error_pct)
    else:
        accuracy = 100.0 if detected == 0 else 0.0

    filename = os.path.basename(img_path)
    print(f"  {'File':<30s} {'GT':>4s} {'Det':>4s} {'Accuracy':>9s}")
    print(f"  {'-'*30} {'-'*4} {'-'*4} {'-'*9}")
    print(f"  {filename:<30s} {ground_truth_count:>4d} {detected:>4d} "
          f"{accuracy:>8.1f}%")

    return detected, accuracy


# ═══════════════════════════════════════════════════════
# FUNCTION 6 — run_batch
# ═══════════════════════════════════════════════════════

def run_batch():
    """
    Batch-process all images found in DATA_DIR.

    Scans for .png, .jpg, .jpeg, .bmp files, runs the pipeline on each,
    generates growth analysis if 2+ images, and saves a JSON summary.
    """
    SUPPORTED = {".png", ".jpg", ".jpeg", ".bmp"}

    # Collect image files
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)

    image_files = sorted(
        f for f in os.listdir(DATA_DIR)
        if os.path.splitext(f)[1].lower() in SUPPORTED
    )

    if not image_files:
        print("\n" + "=" * 50)
        print("  No images found in the 'data/' folder.")
        print("=" * 50)
        print("\n  To get started:")
        print("    1. Place your plant images in the 'data/' folder")
        print("       Supported formats: .png  .jpg  .jpeg  .bmp")
        print("    2. Run this script again:  python main.py")
        print("    3. Or run demo.py first to test with synthetic images\n")
        # Save empty summary so the JSON file always exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
            json.dump([], f, indent=2)
        return

    print("\n" + "=" * 56)
    print("  Greenhouse Crop Growth Monitoring -- Batch Processing")
    print("=" * 56)
    print(f"\n  Found {len(image_files)} image(s) in '{DATA_DIR}/'\n")

    results = []
    paths   = []

    for fname in image_files:
        path = os.path.join(DATA_DIR, fname)
        paths.append(path)
        print(f"  Processing: {fname}")
        try:
            n = visualize_pipeline(path, save=True)
            results.append({"file": fname, "plants_detected": n})
            print(f"    -> {n} plant(s) detected\n")
        except Exception as e:
            print(f"    ERROR: {e}\n")
            results.append({"file": fname, "plants_detected": -1})

    # Growth analysis if we have enough images
    if len(paths) >= 2:
        print("  Running growth analysis (treating images as time points)...\n")
        growth = measure_growth(paths)
        print()
        plot_growth(growth, save=True)

    # Save JSON summary
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    try:
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved: {summary_path}")
    except OSError as e:
        print(f"  ERROR saving summary: {e}")

    # Final table
    print("\n  " + "-" * 40)
    print(f"  {'File':<30s} {'Plants':>6s}")
    print("  " + "-" * 40)
    for r in results:
        det = r["plants_detected"]
        det_str = str(det) if det >= 0 else "ERROR"
        print(f"  {r['file']:<30s} {det_str:>6s}")
    print("  " + "-" * 40)
    print(f"  Total images processed: {len(results)}")
    print()


# ═══════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    run_batch()
