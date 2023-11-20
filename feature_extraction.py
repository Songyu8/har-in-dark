import os
import cv2
import clip
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
from tqdm import tqdm

def uniform_sampling_save_images(video_path, output_folder, interval):
    cap = cv2.VideoCapture(video_path)
    current_frame = 0

    # 创建输出文件夹

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame % interval == 0:
            output_path = os.path.join(output_folder, f"frame_{current_frame}.png")
            cv2.imwrite(output_path, frame)

        current_frame += 1

    cap.release()

# 1. 定义文件路径
txt_file_path = "C:\\Users\\songyu\\Desktop\\EE6222 train and validate 2023\\validate.txt"  # 替换为你的实际文件路径
base_folder = "C:\\Users\\songyu\\Desktop\\EE6222 train and validate 2023\\validate"

# 2. 定义预训练模型和变换
# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.07, 0.07, 0.07], std=[0.1, 0.09, 0.08]),
])

# 3. 读取文本文件并进行特征提取
with open(txt_file_path, 'r') as file:
    lines = file.readlines()

for line in tqdm(lines):
    # 按制表符分隔每行数据
    data = line.strip().split('\t')
    
    # 获取视频类别、子文件夹和文件名
    category = int(data[1])
    # subfolder, file_name = data[2].split('/')
    file_name=data[2]
    
    # 构建完整的视频文件路径
    # video_path = os.path.join(base_folder, subfolder, file_name)

    video_path = os.path.join(base_folder,file_name)

    
    # 处理视频文件，进行特征提取
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    current_frame = 0
    
    all_features = []
    all_labels = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame % 2 == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = transform(frame)
            frame = torch.unsqueeze(frame, 0)

            with torch.no_grad():
                    features = model.encode_image(frame.to(device))
                    all_features.append(features)
                    all_labels.append(category)

        current_frame += 1


    cap.release()

    # 特征融合
    fused_feature = torch.mean(torch.stack(all_features), dim=0).cpu()

    # 提取视频文件名（去除扩展名）
    video_name = os.path.splitext(file_name)[0]

    # 保存特征
    # save_path = os.path.join(base_folder, subfolder, f"{video_name}_features.npy")
    save_path = os.path.join(base_folder,  f"{video_name}_features.npy")
    np.save(save_path, fused_feature)

# print("特征提取完成。")