"""
growth_analysis.py — Custom growth analysis & accuracy checking
================================================================
Use this script when you have your OWN dataset of plant images.

HOW TO USE:
  1. Put your images in the data/ folder.
  2. Edit IMAGE_SEQUENCE below to list your images in time order.
  3. Edit GROUND_TRUTH below to add known plant counts (optional).
  4. Run:  python growth_analysis.py

The script will:
  - Measure plant area for each image and plot growth over time
  - Evaluate accuracy against ground truth (if provided)
"""

import os
import sys

# Import the core pipeline functions
from main import (
    measure_growth,
    plot_growth,
    check_accuracy,
    visualize_pipeline,
    DATA_DIR,
    OUTPUT_DIR,
)


# ═══════════════════════════════════════════════════════
# CONFIGURATION — EDIT THESE FOR YOUR DATASET
# ═══════════════════════════════════════════════════════

# List your images in chronological order for growth analysis.
# Each entry: (path_to_image, label_for_chart)
# Uncomment and modify the examples below:
IMAGE_SEQUENCE = [
    # ("data/plant_day01.png", "Day 1"),
    # ("data/plant_day07.png", "Day 7"),
    # ("data/plant_day14.png", "Day 14"),
    # ("data/plant_day21.png", "Day 21"),
]

# Ground truth plant counts for accuracy evaluation.
# Key = filename (just the name, not the full path)
# Value = known number of plants in that image
# Uncomment and modify:
GROUND_TRUTH = {
    # "plant001_rgb.png": 5,
    # "plant002_rgb.png": 3,
    # "plant003_rgb.png": 7,
}


# ═══════════════════════════════════════════════════════
# MAIN LOGIC
# ═══════════════════════════════════════════════════════

def main():
    """Run growth analysis and/or accuracy evaluation."""

    print("=" * 56)
    print("  Growth Analysis & Accuracy Evaluation")
    print("=" * 56)

    ran_something = False

    # ── Growth analysis ──────────────────────────────────
    if IMAGE_SEQUENCE:
        print("\n[Growth Analysis]\n")
        paths  = [entry[0] for entry in IMAGE_SEQUENCE]
        labels = [entry[1] for entry in IMAGE_SEQUENCE]

        # Verify all files exist before processing
        missing = [p for p in paths if not os.path.isfile(p)]
        if missing:
            print("  ERROR: The following images were not found:")
            for m in missing:
                print(f"    - {m}")
            print("\n  Please check the paths in IMAGE_SEQUENCE.\n")
        else:
            # Run pipeline visualisation for each image
            for path in paths:
                print(f"  Processing: {path}")
                try:
                    n = visualize_pipeline(path, save=True)
                    print(f"    -> {n} plant(s) detected\n")
                except Exception as e:
                    print(f"    ERROR: {e}\n")

            # Measure growth over time
            print("  Measuring plant growth...\n")
            growth = measure_growth(paths, labels=labels)

            if len(growth) >= 2:
                print("\n  Plotting growth chart...\n")
                plot_growth(growth, save=True)

            ran_something = True
    else:
        print("\n  IMAGE_SEQUENCE is empty.")
        print("  To use growth analysis:")
        print("    1. Open growth_analysis.py in a text editor")
        print("    2. Add your images to the IMAGE_SEQUENCE list")
        print("    3. Format: (\"data/your_image.png\", \"Label\")")
        print("    4. Run again: python growth_analysis.py\n")

    # ── Accuracy evaluation ──────────────────────────────
    if GROUND_TRUTH:
        print("\n[Accuracy Evaluation]\n")
        accuracies = []

        for filename, gt_count in GROUND_TRUTH.items():
            path = os.path.join(DATA_DIR, filename)
            if not os.path.isfile(path):
                print(f"  WARNING: '{path}' not found, skipping.\n")
                continue
            try:
                detected, acc = check_accuracy(path, gt_count)
                accuracies.append(acc)
                print()
            except Exception as e:
                print(f"  ERROR on '{filename}': {e}\n")

        # Print average accuracy
        if accuracies:
            avg = sum(accuracies) / len(accuracies)
            print(f"  {'='*40}")
            print(f"  Average accuracy across {len(accuracies)} image(s): "
                  f"{avg:.1f}%")
            print(f"  {'='*40}\n")

        ran_something = True
    else:
        if not IMAGE_SEQUENCE:
            print("  GROUND_TRUTH is also empty.")
            print("  To evaluate accuracy:")
            print("    1. Open growth_analysis.py in a text editor")
            print("    2. Add entries to the GROUND_TRUTH dictionary")
            print("    3. Format: \"filename.png\": count")
            print("    4. Run again: python growth_analysis.py\n")

    if not ran_something:
        print("  -" * 28)
        print("  Nothing to do. Edit this file to configure your dataset.")
        print("  Or run demo.py first to see the pipeline in action.\n")


if __name__ == "__main__":
    main()
