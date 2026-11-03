import os
import cv2
# in search(shift + command + p) : >python: interepter


def verify_yolo_labels(image_dir, label_dir, class_names=None):
  """Reads images and their corresponding converted YOLO .txt label files,

  then draws bounding boxes using OpenCV to verify accuracy.
  """
  if not os.path.exists(image_dir):
    print(f"Image directory not found: {image_dir}")
    return

  if not os.path.exists(label_dir):
    print(f"Label directory not found: {label_dir}")
    return

  # Default class names based on your COCO categories
  if class_names is None:
    class_names = {0: "person", 1: "car", 2: "street lamp"}

  # Loop through all files in the image directory
  for filename in os.listdir(image_dir):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
      continue

    img_path = os.path.join(image_dir, filename)
    base_name = os.path.splitext(filename)[0]
    txt_path = os.path.join(label_dir, base_name + ".txt")

    # Read the image
    img = cv2.imread(img_path)
    if img is None:
      print(f"Could not read image: {img_path}")
      continue

    h_img, w_img, _ = img.shape

    # Check if corresponding YOLO text label file exists
    if not os.path.exists(txt_path):
      print(f"No label file found for {filename}, skipping visualization.")
      continue

    # Read YOLO labels
    with open(txt_path, "r", encoding="utf-8") as f:
      lines = f.readlines()

    for line in lines:
      parts = line.strip().split()
      if not parts:
        continue

      class_id = int(parts[0])
      x_center = float(parts[1])
      y_center = float(parts[2])
      w_norm = float(parts[3])
      h_norm = float(parts[4])

      # Convert normalized YOLO coordinates back to absolute OpenCV pixel coordinates
      w_box = w_norm * w_img
      h_box = h_norm * h_img
      x_min = int((x_center * w_img) - (w_box / 2.0))
      y_min = int((y_center * h_img) - (h_box / 2.0))
      x_max = int(x_min + w_box)
      y_max = int(y_min + h_box)

      # Draw the bounding box
      color = (0, 255, 0)  # Green box
      cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

      # Add class label text
      label_text = class_names.get(class_id, f"class_{class_id}")
      cv2.putText(
          img,
          label_text,
          (x_min, max(20, y_min - 5)),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.5,
          (0, 0, 255),
          2,
      )

    # Show the image window (Press any key to move to the next image, 'q' to quit)
    cv2.imshow(f"Verification - {filename}", img)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

    if key == ord("q"):
      break


# Example usage:
verify_yolo_labels('images/', 'yolo_labels/')