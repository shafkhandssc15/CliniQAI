import os
import io
import json
import pandas as pd
from PIL import Image

def download_dataset():
    print("Starting download of real-world dataset from Hugging Face...")
    parquet_url = "https://huggingface.co/datasets/davanstrien/india-medical-ocr-test/resolve/main/data/train-00000-of-00001.parquet"
    
    # Create the output directory
    output_dir = os.path.join("data", "real_reports")
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the parquet file
    print("Fetching parquet file...")
    df = pd.read_parquet(parquet_url)
    print(f"Dataset loaded. Total records available: {len(df)}")
    
    metadata = {}
    
    # We will download the first 5 report files to serve as our local benchmark suite
    num_samples = min(5, len(df))
    print(f"Extracting first {num_samples} samples...")
    
    for idx in range(num_samples):
        row = df.iloc[idx]
        image_data = row['image']
        markdown_text = row['markdown']
        
        # Save image
        img_name = f"real_report_{idx}.png"
        img_path = os.path.join(output_dir, img_name)
        
        try:
            # Handle different image representations in Hugging Face Datasets Parquet
            if isinstance(image_data, dict) and 'bytes' in image_data:
                # Binary bytes dict
                img = Image.open(io.BytesIO(image_data['bytes']))
            elif isinstance(image_data, bytes):
                # Raw bytes
                img = Image.open(io.BytesIO(image_data))
            elif isinstance(image_data, dict) and 'path' in image_data:
                # If it's a dictionary but bytes are missing, try PIL opening if path is loadable (unlikely from remote parquet)
                print(f"Warning: Image at index {idx} has path but no bytes. Skipping.")
                continue
            else:
                # It might be loaded directly as a PIL Image object by pandas/pyarrow
                img = image_data
                if not isinstance(img, Image.Image):
                    # Try raw conversion
                    img = Image.open(io.BytesIO(bytes(image_data)))
            
            # Save the image as PNG
            img.save(img_path, format="PNG")
            
            # Save the ground truth markdown
            gt_name = f"real_report_{idx}_gt.md"
            gt_path = os.path.join(output_dir, gt_name)
            with open(gt_path, "w", encoding="utf-8") as f:
                f.write(markdown_text)
                
            metadata[img_name] = {
                "ground_truth_file": gt_name,
                "label": "Real Historical Lab Report",
                "source": "india-medical-ocr-test"
            }
            print(f"Successfully extracted {img_name}")
            
        except Exception as e:
            print(f"Error extracting record at index {idx}: {e}")
            
    # Save the consolidated metadata file
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Real-world benchmark dataset setup complete! Metadata saved to {metadata_path}")

if __name__ == "__main__":
    download_dataset()
