import os
import sys

# Import the synthetic image generator from demo.py
from demo import generate_synthetic_image
from main import DATA_DIR

def generate_bulk_images(num_images=100):
    print("=" * 56)
    print(f"  Generating {num_images} Synthetic Crop Images")
    print("=" * 56)
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    for i in range(1, num_images + 1):
        # Format the filename, e.g., demo_day001.png
        filename = f"dataset_day{i:03d}.png"
        
        # Simulate gradual growth over 100 days
        # Number of plants slowly increases from 3 up to around 15
        n_plants = 3 + int((i / num_images) * 12)
        
        # Base size slowly increases
        min_size = 18 + int((i / num_images) * 40)
        max_size = 28 + int((i / num_images) * 50)
        
        # Unique seed for reproducibility based on the day
        seed = 42 + i
        
        generate_synthetic_image(filename, n_plants, (min_size, max_size), seed)
        
    print("\n" + "=" * 56)
    print(f"  Successfully generated {num_images} images in '{DATA_DIR}/'")
    print("=" * 56 + "\n")

if __name__ == "__main__":
    generate_bulk_images(100)
