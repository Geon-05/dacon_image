import os
from PIL import Image
import torch
import cv2
import numpy as np

def mask_function(input_path, gt_path):
    try:
        # Load and preprocess images
        input_image = Image.open(input_path).convert("RGB")  # Ensure RGB format
        input_image_np = np.array(input_image)
        gt_image_gray = Image.open(gt_path).convert("L")  # Load mask as grayscale
        gt_image_gray_np = np.array(gt_image_gray)

        # Convert input_image_np to grayscale
        input_image_gray_np = cv2.cvtColor(input_image_np, cv2.COLOR_RGB2GRAY)

        # Compute the difference
        difference = cv2.absdiff(gt_image_gray_np, input_image_gray_np)

        # Threshold the difference to create a binary mask
        _, binary_difference = cv2.threshold(difference, 1, 255, cv2.THRESH_BINARY)

        # Remove small noise with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        binary_difference = cv2.morphologyEx(binary_difference, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(binary_difference, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Fill the contours to create a mask
        mask_filled = np.zeros_like(binary_difference)
        cv2.drawContours(mask_filled, contours, -1, color=255, thickness=cv2.FILLED)

        # Expand the filled mask (dilation)
        mask_filled = cv2.dilate(mask_filled, kernel, iterations=1)

        # Convert input image and mask to PyTorch tensors
        mask_tensor = torch.tensor(mask_filled, dtype=torch.float32).unsqueeze(0) / 255.0  # Normalize mask to [0, 1]

        return mask_tensor
    except Exception as e:
        print(f"Mask creation failed for input: {input_path}, error: {e}")
        return torch.zeros((1, 512, 512), dtype=torch.float32)


# 손상 이미지 폴더, 정답 이미지 폴더, 출력 마스크 폴더 설정
input_images_dir = "train_input"
gt_images_dir = "train_gt"
output_masks_dir = "output_masks"

os.makedirs(output_masks_dir, exist_ok=True)

# 손상 이미지 폴더에 있는 모든 이미지 파일 이름 리스트 획득
input_image_files = sorted([f for f in os.listdir(input_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

for filename in input_image_files:
    input_path = os.path.join(input_images_dir, filename)
    gt_path = os.path.join(gt_images_dir, filename)

    # 마스크 생성
    mask_tensor = mask_function(input_path, gt_path)

    # mask_tensor는 shape [1, H, W]의 텐서이며 값 범위 [0,1]
    # 이를 이미지로 저장하기 위해 numpy로 변환 후 0~255 범위로 스케일링
    mask_np = (mask_tensor.squeeze(0).numpy() * 255).astype(np.uint8)

    # 마스크 이미지 저장
    output_path = os.path.join(output_masks_dir, filename)
    mask_img = Image.fromarray(mask_np)
    mask_img.save(output_path)

    print(f"Saved mask for {filename} at {output_path}")
