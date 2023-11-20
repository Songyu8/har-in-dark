# std
import argparse
from argparse import RawTextHelpFormatter
import glob
import os
from os import makedirs
from os.path import join, exists, basename, splitext
# 3p
import cv2
from tqdm import tqdm
# project
from exposure_enhancement import enhance_image_exposure

def main(args):

    input_folder = "C:\\Users\\songyu\\Desktop\\output_images_test"
    output_folder = "C:\\Users\\songyu\\Desktop\\output_images_enhance_test"

    # 检查输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 递归获取目录中所有文件
    for root, dirs, files in tqdm(os.walk(input_folder),desc="Enhancing images"):
        for file in tqdm(files):
            # 过滤出具有特定扩展名（例如 '.png'）的文件
            if file.endswith('.png'):
                # 构建图片的完整路径
                image_path = os.path.join(root, file)



                image = cv2.imread(image_path)

                image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                data=root.split('\\')


                save_folder=os.path.join(output_folder,data[-1])

                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)

                save_path=os.path.join(save_folder,file)


                enhanced_image = enhance_image_exposure(image, args.gamma, args.lambda_, not args.lime,
                                sigma=args.sigma, bc=args.bc, bs=args.bs, be=args.be, eps=args.eps)

                cv2.imwrite(save_path, enhanced_image)

    
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Python implementation of two low-light image enhancement techniques via illumination map estimation.",
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument("-f", '--folder', default='C:\\Users\\songyu\\Desktop\\Low-light-Image-Enhancement-master\\EE6222 train and validate 2023\\train', type=str,
                        help="folder path to test images.")
    parser.add_argument("-g", '--gamma', default=0.6, type=float,
                        help="the gamma correction parameter.")
    parser.add_argument("-l", '--lambda_', default=0.15, type=float,
                        help="the weight for balancing the two terms in the illumination refinement optimization objective.")
    parser.add_argument("-ul", "--lime", action='store_true',
                        help="Use the LIME method. By default, the DUAL method is used.")
    parser.add_argument("-s", '--sigma', default=3, type=int,
                        help="Spatial standard deviation for spatial affinity based Gaussian weights.")
    parser.add_argument("-bc", default=1, type=float,
                        help="parameter for controlling the influence of Mertens's contrast measure.")
    parser.add_argument("-bs", default=1, type=float,
                        help="parameter for controlling the influence of Mertens's saturation measure.")
    parser.add_argument("-be", default=1, type=float,
                        help="parameter for controlling the influence of Mertens's well exposedness measure.")
    parser.add_argument("-eps", default=1e-3, type=float,
                        help="constant to avoid computation instability.")

    args = parser.parse_args()
    main(args)
