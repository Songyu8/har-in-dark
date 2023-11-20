import cv2
import os
import random
from tqdm import tqdm


# 1. 定义文件路径
txt_file_path = "C:\\Users\\songyu\\Desktop\\EE6222 train and validate 2023\\validate.txt"  # 替换为你的实际文件路径
base_folder = "C:\\Users\\songyu\\Desktop\\EE6222 train and validate 2023\\validate"
output_folder = 'C:\\Users\\songyu\\Desktop\\output_images_test'



with open(txt_file_path, 'r') as file:
    lines = file.readlines()

for line in tqdm(lines):
    # 按制表符分隔每行数据
    data = line.strip().split('\t')

    # print(data)

    # # 获取视频类别、子文件夹和文件名
    # category = int(data[1])
    # subfolder, file_name = data[2].split('/')

    # # 构建完整的视频文件路径
    video_path = os.path.join(base_folder, data[2])

    current_frame = 0

    # video_name_without_extension = os.path.splitext(file_name)[0]

    # # 处理视频文件，进行特征提取
    cap = cv2.VideoCapture(video_path)

    exten=data[2].split('.')

    output_image_folder=os.path.join(output_folder,exten[0])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if current_frame % 2 == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # # enhance images

            output_path = os.path.join(output_image_folder,f"frame_{current_frame}.png")

            print(output_path)


            if not os.path.exists(output_image_folder):
                os.makedirs(output_image_folder)

            cv2.imwrite(output_path ,frame)
        current_frame+=1