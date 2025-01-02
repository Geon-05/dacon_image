import os
import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################
# 모델 구조 정의 (학습시 사용한 것과 동일)
############################

class GatedConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=torch.nn.ReLU()):
        super(GatedConv2d, self).__init__()
        self.feature_conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.mask_conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.sigmoid = torch.nn.Sigmoid()
        self.activation = activation

    def forward(self, x):
        f = self.feature_conv(x)
        m = self.mask_conv(x)
        gated = self.sigmoid(m)
        if self.activation is not None:
            f = self.activation(f)
        return f * gated

class GatedDeconv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, activation=torch.nn.ReLU()):
        super(GatedDeconv2d, self).__init__()
        self.feature_deconv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.mask_deconv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.sigmoid = torch.nn.Sigmoid()
        self.activation = activation

    def forward(self, x):
        f = self.feature_deconv(x)
        m = self.mask_deconv(x)
        gated = self.sigmoid(m)
        if self.activation is not None:
            f = self.activation(f)
        return f * gated

class ContextualAttention(torch.nn.Module):
    def __init__(self, kernel_size=3, stride=1, dilation=1):
        super(ContextualAttention, self).__init__()
        self.conv = torch.nn.Conv2d(512, 512, kernel_size, stride, dilation, bias=False)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        B,C,H,W = x.size()
        query = x.view(B,C,-1)
        key = x.view(B,C,-1)
        value = x.view(B,C,-1)
        attn = torch.bmm(query.permute(0,2,1), key)
        attn = self.softmax(attn)
        out = torch.bmm(attn, value.permute(0,2,1))
        out = out.permute(0,2,1).view(B,C,H,W)
        out = self.conv(out)
        return out

class Stage1Generator(torch.nn.Module):
    def __init__(self):
        super(Stage1Generator, self).__init__()
        self.encoder = torch.nn.Sequential(
            GatedConv2d(4, 64, 4, 2, 1),
            GatedConv2d(64, 128, 4, 2, 1),
            GatedConv2d(128, 256, 4, 2, 1),
            GatedConv2d(256, 512, 4, 2, 1)
        )
        self.decoder = torch.nn.Sequential(
            GatedDeconv2d(512, 256, 4, 2, 1),
            GatedDeconv2d(256, 128, 4, 2, 1),
            GatedDeconv2d(128, 64, 4, 2, 1),
            GatedDeconv2d(64, 64, 4, 2, 1, activation=torch.nn.ReLU()),
            torch.nn.Conv2d(64, 3, 3, 1, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x, mask):
        inp = torch.cat((x, mask), dim=1)
        feat = self.encoder(inp)
        out = self.decoder(feat)
        return out

class Stage2Generator(torch.nn.Module):
    def __init__(self):
        super(Stage2Generator, self).__init__()
        self.encoder = torch.nn.Sequential(
            GatedConv2d(7, 64, 4, 2, 1),
            GatedConv2d(64, 128, 4, 2, 1),
            GatedConv2d(128, 256, 4, 2, 1),
            GatedConv2d(256, 512, 4, 2, 1)
        )
        self.contextual_attention = ContextualAttention()
        self.decoder = torch.nn.Sequential(
            GatedDeconv2d(512, 256, 4, 2, 1),
            GatedDeconv2d(256, 128, 4, 2, 1),
            GatedDeconv2d(128, 64, 4, 2, 1),
            GatedDeconv2d(64, 64, 4, 2, 1, activation=torch.nn.ReLU()),
            torch.nn.Conv2d(64, 3, 3, 1, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, coarse_out, inp, mask):
        fin_inp = torch.cat((coarse_out, inp, mask), dim=1)
        feat = self.encoder(fin_inp)
        feat = self.contextual_attention(feat)
        out = self.decoder(feat)
        return out

# 최적 모델 가중치 경로
best_coarse_path = "best_coarse_generator_epoch4"
best_fine_path = "best_fine_generator_epoch4.pth"


# 테스트 이미지(손상된 컬러 이미지) 폴더
test_input_dir = r"03_damage/data/output_colToper_2024123002"
test_mask_dir = r"data\output_01_mask"
output_dir = "output_colToper"

os.makedirs(output_dir, exist_ok=True)

coarse_generator = Stage1Generator().to(device)
fine_generator = Stage2Generator().to(device)

coarse_generator.load_state_dict(torch.load(best_coarse_path, map_location=device, weights_only=True))
fine_generator.load_state_dict(torch.load(best_fine_path, map_location=device, weights_only=True))

coarse_generator.eval()
fine_generator.eval()

transform = T.Compose([
    T.Resize((512,512)),
    T.ToTensor()
])

test_files = [f for f in os.listdir(test_input_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]

with torch.no_grad():
    for filename in test_files:
        input_path = os.path.join(test_input_dir, filename)
        mask_path = os.path.join(test_mask_dir, filename)

        if not os.path.exists(mask_path):
            print(f"{mask_path}가 존재하지 않습니다. 스킵합니다.")
            continue

        inp_img = Image.open(input_path).convert("RGB")
        mask_img = Image.open(mask_path).convert("L")

        inp_tensor = transform(inp_img).unsqueeze(0).to(device)   # (1,3,H,W)
        mask_tensor = transform(mask_img).unsqueeze(0).to(device) # (1,1,H,W)

        # 손상 영역 0 처리
        mask_broadcast = mask_tensor.expand_as(inp_tensor)
        damaged_inp = inp_tensor * (1 - mask_broadcast)

        # 복원
        coarse_out = coarse_generator(damaged_inp, mask_tensor)
        fine_out = fine_generator(coarse_out, damaged_inp, mask_tensor)

        # 여기서 복원된 부분(fine_out)을 마스크가 1인 영역에만 적용
        # final_result = original_damaged_image * (1 - mask) + fine_out * mask
        final_result = inp_tensor * (1 - mask_broadcast) + fine_out * mask_broadcast

        # 결과 텐서를 이미지로 변환
        final_result_pil = T.ToPILImage()(final_result.squeeze(0).cpu())

        save_path = os.path.join(output_dir, filename)
        final_result_pil.save(save_path)
        print(f"{filename} 복원 완료(마스크 영역만 덮어쓰기) -> {save_path}")
