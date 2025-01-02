import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms, models
from torchvision.models import resnet50, ResNet50_Weights

#------------- 모델 정의 부분 (학습 시 사용한 코드와 동일해야 함) -------------#
def rgb_to_lab_normalized(rgb):
    rgb_np = (rgb.permute(1,2,0).numpy() * 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    L = lab[:,:,0] / 255.0
    a = (lab[:,:,1] - 128.0)/128.0
    b = (lab[:,:,2] - 128.0)/128.0
    return L, a, b

def lab_to_rgb(L, a, b):
    lab_0_255 = np.zeros((L.shape[0], L.shape[1], 3), dtype=np.float32)
    lab_0_255[:,:,0] = L * 255.0
    lab_0_255[:,:,1] = a * 128.0 + 128.0
    lab_0_255[:,:,2] = b * 128.0 + 128.0

    lab_0_255 = np.clip(lab_0_255, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(lab_0_255, cv2.COLOR_Lab2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = np.clip(rgb,0,255).astype(np.uint8)
    return rgb / 255.0

class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpConv, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
    def forward(self, x):
        return self.up(x)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class ResNetEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        net = models.resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        self.initial = nn.Sequential(net.conv1, net.bn1, net.relu)
        self.maxpool = net.maxpool
        self.layer1 = net.layer1 # 256채널
        self.layer2 = net.layer2 # 512채널
        self.layer3 = net.layer3 # 1024채널
        self.layer4 = net.layer4 # 2048채널

    def forward(self, x):
        x0 = self.initial(x)   #64채널
        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)   #256
        x2 = self.layer2(x1)   #512
        x3 = self.layer3(x2)   #1024
        x4 = self.layer4(x3)   #2048
        return x0, x1, x2, x3, x4

class ResNetUNet(nn.Module):
    def __init__(self, out_ch=2, pretrained=True):
        super().__init__()
        self.encoder = ResNetEncoder(pretrained=pretrained)
        self.up3 = UpConv(2048, 1024)
        self.dec3 = DoubleConv(2048, 1024)

        self.up2 = UpConv(1024, 512)
        self.dec2 = DoubleConv(1024, 512)

        self.up1 = UpConv(512, 256)
        self.dec1 = DoubleConv(512, 256)

        self.up0 = UpConv(256, 64)
        self.dec0 = DoubleConv(128, 64)

        self.up_final = UpConv(64,64)
        self.dec_final = DoubleConv(64,64)
        self.final_out = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        x = x.repeat(1,3,1,1)
        x0, x1, x2, x3, x4 = self.encoder(x)

        x_up3 = self.up3(x4)           
        x_cat3 = torch.cat([x_up3, x3], dim=1) 
        x_dec3 = self.dec3(x_cat3)     

        x_up2 = self.up2(x_dec3)       
        x_cat2 = torch.cat([x_up2, x2], dim=1)
        x_dec2 = self.dec2(x_cat2)     

        x_up1 = self.up1(x_dec2)       
        x_cat1 = torch.cat([x_up1, x1], dim=1)
        x_dec1 = self.dec1(x_cat1)     

        x_up0 = self.up0(x_dec1)       
        x_cat0 = torch.cat([x_up0, x0], dim=1)
        x_dec0 = self.dec0(x_cat0)

        x_upf = self.up_final(x_dec0)
        x_decf = self.dec_final(x_upf)
        out = self.final_out(x_decf)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로드
model = ResNetUNet(out_ch=2, pretrained=False).to(device)
model.load_state_dict(torch.load("/home/zqrc05/project/imagepro/test/model/grayTocol/12_05_best_model_7_0.043.pth", map_location=device))
model.eval()

# 테스트 디렉토리 지정
test_gray_dir = "/home/zqrc05/project/imagepro/test/test_input"  # 손상된 흑백 이미지들이 있는 폴더 경로
test_mask_dir = "/home/zqrc05/project/imagepro/test/mask"  # 마스크 이미지들이 있는 폴더 경로
output_dir = "/home/zqrc05/project/imagepro/test/output_grayTocol"     # 복원 결과를 저장할 폴더 경로

os.makedirs(output_dir, exist_ok=True)

transform_gray = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor()
])

# test_gray_dir 내 모든 png 이미지 처리
gray_image_paths = sorted(glob.glob(os.path.join(test_gray_dir, "*.png")))

with torch.no_grad():
    for gray_path in gray_image_paths:
        fname = os.path.basename(gray_path)  # 예: TEST_001.png
        mask_path = os.path.join(test_mask_dir, fname) # 동일 이름의 mask

        if not os.path.exists(mask_path):
            print(f"No matching mask found for {fname}, skipping...")
            continue

        # 흑백 이미지 로드
        gray_img = Image.open(gray_path).convert('L')
        gray_tensor = transform_gray(gray_img) # [1,H,W]

        # 마스크 로드
        mask_img = Image.open(mask_path).convert('L')
        mask_np = np.array(mask_img)
        mask_bin = (mask_np > 128).astype(np.float32)
        mask_bin = torch.from_numpy(mask_bin).unsqueeze(0) # [1,H,W]
        # 사이즈가 다르다면 아래 주석 제거 후 interpolate 적용
        # mask_bin = F.interpolate(mask_bin.unsqueeze(0), size=(512,512), mode='bilinear', align_corners=False).squeeze(0)

        gray_tensor = gray_tensor.unsqueeze(0).to(device) # [1,1,H,W]
        mask_bin = mask_bin.to(device) # [1,H,W]

        # 모델 추론
        pred_ab = model(gray_tensor) # [1,2,H,W]

        # 결과 복원
        pred_ab_np = pred_ab[0].cpu().permute(1,2,0).numpy()  # [H,W,2]
        L_np = gray_tensor[0,0].cpu().numpy() # [H,W]

        # Lab->RGB 복원 (L은 gray에서 가져옴)
        pred_rgb = lab_to_rgb(L_np, pred_ab_np[:,:,0], pred_ab_np[:,:,1])

        # 결과 저장
        out_path = os.path.join(output_dir, fname)
        Image.fromarray((pred_rgb*255).astype(np.uint8)).save(out_path)
        print(f"Saved restored image: {out_path}")
