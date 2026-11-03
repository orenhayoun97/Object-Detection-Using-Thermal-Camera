import json
import os
import random
import shutil


def process_coco_to_yolo_dataset(
    json_file, images_dir, output_dir, train_ratio=0.7, val_ratio=0.15
):
  """Converts a COCO dataset into a YOLO-formatted

  randomized train/val/test splits.

  Parameters:
  - json_file (str): Path to the COCO JSON annotation file.
  - images_dir (str): Directory containing the source images.
  - output_dir (str): Main output directory for the YOLO dataset.
  - train_ratio (float): Proportion of data for training (default 0.7).
  - val_ratio (float): Proportion of data for validation (default 0.15).
  """
  # Ensure paths are absolute if needed
  if not os.path.isabs(json_file):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_file = os.path.join(script_dir, json_file)

  if not os.path.isabs(images_dir):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(script_dir, images_dir)

  with open(json_file, "r", encoding="utf-8") as f:
    coco_data = json.load(f)

  # 1. Create Directory Structure
  splits = ["train", "val", "test"]
  for split in splits:
    os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

  # 2. Map Categories (adapted from source 1 and source 2)[cite: 1, 2]
  sorted_categories = sorted(coco_data["categories"], key=lambda x: x["id"])
  category_mapping = {cat["id"]: idx for idx, cat in enumerate(sorted_categories)}
  category_names = [cat["name"] for cat in sorted_categories]
  nc = len(category_names)

  # Map image_id to image details
  images_dict = {}
  for img in coco_data["images"]:
    images_dict[img["id"]] = {
        "file_name": img["file_name"],
        "width": img["width"],
        "height": img["height"],
    }

  # Group annotations by image_id (adapted from source 2)[cite: 2]
  annotations_dict = {}
  for ann in coco_data["annotations"]:
    img_id = ann["image_id"]
    if img_id not in annotations_dict:
      annotations_dict[img_id] = []
    annotations_dict[img_id].append(ann)

  # Get list of all valid image IDs that exist in the source directory
  image_ids = []
  for img_id, img_info in images_dict.items():
    src_img_path = os.path.join(images_dir, img_info["file_name"])
    if os.path.exists(src_img_path):
      image_ids.append(img_id)

  # 3. Randomized Dataset Splitting (Ensures a different split on each execution)
  random.seed(None)  # Uses system time for dynamic randomization
  random.shuffle(image_ids)

  total_images = len(image_ids)
  train_end = int(total_images * train_ratio)
  val_end = train_end + int(total_images * val_ratio)

  train_ids = image_ids[:train_end]
  val_ids = image_ids[train_end:val_end]
  test_ids = image_ids[val_end:]

  split_mapping = {
      "train": train_ids,
      "val": val_ids,
      "test": test_ids,
  }

  # Process and distribute files into respective split folders
  for split_name, ids in split_mapping.items():
    for img_id in ids:
      img_info = images_dict[img_id]
      file_name = img_info["file_name"]
      img_width = img_info["width"]
      img_height = img_info["height"]

      # Copy Image File
      src_img_path = os.path.join(images_dir, file_name)
      dst_img_path = os.path.join(
          output_dir, "images", split_name, os.path.basename(file_name)
      )
      if os.path.exists(src_img_path):
        shutil.copy(src_img_path, dst_img_path)

      # Generate YOLO Bounding Box Annotations (adapted from source 2)[cite: 2]
      txt_filename = os.path.splitext(os.path.basename(file_name))[0] + ".txt"
      txt_path = os.path.join(output_dir, "labels", split_name, txt_filename)

      lines = []
      if img_id in annotations_dict:
        for ann in annotations_dict[img_id]:
          cat_id = ann["category_id"]
          yolo_class_id = category_mapping[cat_id]

          bbox = ann["bbox"]
          xmin, ymin, w, h = bbox

          # Normalize coordinates
          x_center = (xmin + w / 2.0) / img_width
          y_center = (ymin + h / 2.0) / img_height
          norm_w = w / img_width
          norm_h = h / img_height

          # Clamp values between 0.0 and 1.0
          x_center = max(0.0, min(1.0, x_center))
          y_center = max(0.0, min(1.0, y_center))
          norm_w = max(0.0, min(1.0, norm_w))
          norm_h = max(0.0, min(1.0, norm_h))

          lines.append(
              f"{yolo_class_id} {x_center:.6f} {y_center:.6f}"
              f" {norm_w:.6f} {norm_h:.6f}"
          )

      with open(txt_path, "w", encoding="utf-8") as txt_file:
        txt_file.write("\n".join(lines))

  # 4. Generate data.yaml configuration file (adapted from source 1)[cite: 1]
  yaml_path = os.path.join(output_dir, "data.yaml")
  yaml_content = f"""path: "{os.path.abspath(output_dir)}"
train: "images/train"
val: "images/val"
test: "images/test"

nc: {nc}
names: {category_names}
"""

  with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(yaml_content)

  print(f"Dataset successfully converted and structured at: {output_dir}")
  print(f"Total processed images: {total_images}")
  print(
      f"Split breakdown -> Train: {len(train_ids)}, Val: {len(val_ids)}, Test:"
      f" {len(test_ids)}"
  )


# Example usage:
if __name__ == "__main__":
  process_coco_to_yolo_dataset(
      json_file="test_day.json",
      images_dir="images",
      output_dir="yolo_day_dataset_output",
      train_ratio=1,
      val_ratio=0,
  )