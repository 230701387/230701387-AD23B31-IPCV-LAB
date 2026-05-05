"""
evaluate.py — Parameter tuning & evaluation suite
===================================================
Tests different DIST_THRESH and MIN_AREA values on all images in data/
to help you find optimal settings for your report.

Usage:
    python evaluate.py

Output:
    output/threshold_comparison.png  — visual comparison of 4 thresholds
    output/evaluation_report.txt     — full text findings

Prerequisites:
    Put at least one image in data/ (or run demo.py first).
"""

import os
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from main import segment_plants, DATA_DIR, OUTPUT_DIR


# Threshold and area values to evaluate
THRESH_VALUES = [0.3, 0.4, 0.5, 0.6, 0.7]
AREA_VALUES   = [100, 200, 500, 1000]
SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


def find_images():
    """
    Scan DATA_DIR for supported image files.

    Returns
    -------
    list[str] — sorted list of full paths
    """
    if not os.path.isdir(DATA_DIR):
        return []
    return sorted(
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXT
    )


def evaluate_thresholds(image_paths):
    """
    Test every DIST_THRESH value on every image and print results.

    Parameters
    ----------
    image_paths : list[str] — paths to images

    Returns
    -------
    results : dict — {thresh: {filename: n_plants}}
    """
    results = {}
    print("\n  DIST_THRESH evaluation")
    print("  " + "-" * 55)
    header = f"  {'File':<28s}"
    for t in THRESH_VALUES:
        header += f"  {t:>5.1f}"
    print(header)
    print("  " + "-" * 55)

    # Initialise results dict
    for t in THRESH_VALUES:
        results[t] = {}

    for path in image_paths:
        fname = os.path.basename(path)
        row = f"  {fname:<28s}"
        for t in THRESH_VALUES:
            try:
                _, n, *_ = segment_plants(path, dist_thresh=t)
                results[t][fname] = n
                row += f"  {n:>5d}"
            except Exception:
                results[t][fname] = -1
                row += f"  {'ERR':>5s}"
        print(row)

    print("  " + "-" * 55)
    return results


def evaluate_min_area(image_paths):
    """
    Test every MIN_AREA value on every image and print results.

    Parameters
    ----------
    image_paths : list[str] — paths to images

    Returns
    -------
    results : dict — {min_area: {filename: n_plants}}
    """
    results = {}
    print("\n  MIN_AREA evaluation")
    print("  " + "-" * 55)
    header = f"  {'File':<28s}"
    for a in AREA_VALUES:
        header += f"  {a:>5d}"
    print(header)
    print("  " + "-" * 55)

    for a in AREA_VALUES:
        results[a] = {}

    for path in image_paths:
        fname = os.path.basename(path)
        row = f"  {fname:<28s}"
        for a in AREA_VALUES:
            try:
                _, n, *_ = segment_plants(path, min_area=a)
                results[a][fname] = n
                row += f"  {n:>5d}"
            except Exception:
                results[a][fname] = -1
                row += f"  {'ERR':>5s}"
        print(row)

    print("  " + "-" * 55)
    return results


def recommend_threshold(thresh_results, image_paths):
    """
    Recommend the best DIST_THRESH based on consistency and count spread.

    Parameters
    ----------
    thresh_results : dict — from evaluate_thresholds()
    image_paths    : list[str]

    Returns
    -------
    best_thresh : float
    """
    # Score each threshold: prefer moderate counts (not too few, not too many)
    scores = {}
    for t in THRESH_VALUES:
        counts = [v for v in thresh_results[t].values() if v >= 0]
        if not counts:
            scores[t] = -1
            continue
        avg = sum(counts) / len(counts)
        # Penalise very high or very low averages
        # Ideal is somewhere in the middle — a balanced segmentation
        variance = sum((c - avg) ** 2 for c in counts) / len(counts)
        # Lower variance across images is better (more consistent)
        # Higher average is generally better (finds more plants)
        scores[t] = avg - 0.5 * (variance ** 0.5)

    best = max(scores, key=lambda k: scores[k])
    return best


def plot_threshold_comparison(image_path):
    """
    Create a 2×2 grid showing the effect of 4 different DIST_THRESH values.

    Parameters
    ----------
    image_path : str — path to the image to use for comparison

    Returns
    -------
    None (saves to output/threshold_comparison.png)
    """
    thresholds_to_show = [0.3, 0.5, 0.6, 0.7]

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    fig.patch.set_facecolor("#0f1117")
    fname = os.path.basename(image_path)
    fig.suptitle(
        f"Effect of DIST_THRESH on Watershed  |  {fname}",
        color="white", fontsize=13, fontweight="bold", y=0.97
    )

    for ax, t in zip(axes.flat, thresholds_to_show):
        try:
            result_rgb, n, *_ = segment_plants(image_path, dist_thresh=t)
            ax.imshow(result_rgb)
            ax.set_title(
                f"DIST_THRESH={t:.1f}  \u2192  {n} plant(s)",
                color="white", fontsize=11, pad=6
            )
        except Exception as e:
            ax.text(0.5, 0.5, f"Error:\n{e}",
                    transform=ax.transAxes, ha="center", va="center",
                    color="red", fontsize=10)
            ax.set_title(f"DIST_THRESH={t:.1f}", color="white", fontsize=11)

        ax.set_facecolor("#0f1117")
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "threshold_comparison.png")
    try:
        fig.savefig(
            out_path, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor()
        )
        print(f"\n  Saved: {out_path}")
    except OSError as e:
        print(f"\n  ERROR saving: {e}")

    plt.close(fig)


def save_evaluation_report(thresh_results, area_results, best_thresh,
                            image_paths):
    """
    Write a plain-text evaluation report to output/evaluation_report.txt.

    Parameters
    ----------
    thresh_results : dict — from evaluate_thresholds()
    area_results   : dict — from evaluate_min_area()
    best_thresh    : float — recommended threshold
    image_paths    : list[str]
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, "evaluation_report.txt")

    lines = []
    lines.append("=" * 60)
    lines.append("  EVALUATION REPORT -- Greenhouse Crop Growth Monitoring")
    lines.append("=" * 60)
    lines.append("")

    # Threshold results
    lines.append("DIST_THRESH EVALUATION")
    lines.append("-" * 55)
    header = f"{'File':<28s}"
    for t in THRESH_VALUES:
        header += f"  {t:>5.1f}"
    lines.append(header)
    lines.append("-" * 55)

    for path in image_paths:
        fname = os.path.basename(path)
        row = f"{fname:<28s}"
        for t in THRESH_VALUES:
            n = thresh_results[t].get(fname, -1)
            row += f"  {n:>5d}" if n >= 0 else f"  {'ERR':>5s}"
        lines.append(row)
    lines.append("-" * 55)
    lines.append(f"\nRecommended DIST_THRESH: {best_thresh:.1f}")
    lines.append("")

    # Area results
    lines.append("MIN_AREA EVALUATION")
    lines.append("-" * 55)
    header = f"{'File':<28s}"
    for a in AREA_VALUES:
        header += f"  {a:>5d}"
    lines.append(header)
    lines.append("-" * 55)

    for path in image_paths:
        fname = os.path.basename(path)
        row = f"{fname:<28s}"
        for a in AREA_VALUES:
            n = area_results[a].get(fname, -1)
            row += f"  {n:>5d}" if n >= 0 else f"  {'ERR':>5s}"
        lines.append(row)
    lines.append("-" * 55)
    lines.append("")

    # Recommendations
    lines.append("RECOMMENDATIONS")
    lines.append("-" * 40)
    lines.append(f"  Best DIST_THRESH: {best_thresh:.1f}")
    lines.append("  - Lower values (0.3) = more aggressive splitting "
                 "(may over-segment)")
    lines.append("  - Higher values (0.7) = more conservative "
                 "(may merge nearby plants)")
    lines.append("  - Default 0.5 works well for most greenhouse images")
    lines.append("")
    lines.append("  MIN_AREA: 200 (default)")
    lines.append("  - Increase to 500+ if detecting too many small noise "
                 "regions")
    lines.append("  - Decrease to 100 for very small seedlings")
    lines.append("")
    lines.append("=" * 60)

    try:
        with open(report_path, "w") as f:
            f.write("\n".join(lines))
        print(f"  Saved: {report_path}")
    except OSError as e:
        print(f"  ERROR saving report: {e}")


def main():
    """Run the full evaluation suite."""
    print("=" * 56)
    print("  Evaluation Suite — Parameter Tuning")
    print("=" * 56)

    # Find images
    image_paths = find_images()
    if not image_paths:
        print("\n  No images found in 'data/' folder.")
        print("  Run demo.py first to create synthetic test images,")
        print("  then re-run this script.\n")
        sys.exit(0)

    print(f"\n  Found {len(image_paths)} image(s) in '{DATA_DIR}/'\n")

    # ── Threshold evaluation ─────────────────────────────
    thresh_results = evaluate_thresholds(image_paths)

    # ── Area evaluation ──────────────────────────────────
    area_results = evaluate_min_area(image_paths)

    # ── Best threshold recommendation ────────────────────
    best = recommend_threshold(thresh_results, image_paths)
    print(f"\n  -> Recommended DIST_THRESH: {best:.1f}")

    # ── Threshold comparison plot ────────────────────────
    print("\n  Generating threshold comparison plot...")
    plot_threshold_comparison(image_paths[0])

    # ── Save text report ─────────────────────────────────
    print("  Generating evaluation report...")
    save_evaluation_report(thresh_results, area_results, best, image_paths)

    print("\n" + "=" * 56)
    print("  Evaluation complete. See output/ for results.")
    print("=" * 56 + "\n")


if __name__ == "__main__":
    main()
