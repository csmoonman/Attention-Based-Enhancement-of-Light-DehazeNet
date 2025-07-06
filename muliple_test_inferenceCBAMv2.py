from inferenceCBAMv2 import image_haze_removel
from PIL import Image
import torchvision
import os
import argparse
import time

def multiple_dehaze_test(directory, gt_directory=None):
    print(f"处理图像目录: {directory}")
    save_dir = "vis_results"
    os.makedirs(save_dir, exist_ok=True)  

    # 获取所有PNG和JPG图像
    file_list = [f for f in os.listdir(directory) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"找到 {len(file_list)} 个图像文件")

    # 初始化指标计算的累积值
    total_metrics = {'psnr': 0, 'ssim': 0, 'mse': 0}
    processed_count = 0

    # 处理每张图像
    for i, filename in enumerate(file_list):
        img_path = os.path.join(directory, filename)
        img = Image.open(img_path)
        
        # 检查是否有对应的Ground Truth图像用于评估
        gt_img = None
        if gt_directory:
            gt_path = os.path.join(gt_directory, filename)
            if os.path.exists(gt_path):
                gt_img = Image.open(gt_path)
        
        # 处理图像
        if img is not None:
            start_time = time.time()
            dehaze_image, metrics = image_haze_removel(img, gt_img)
            process_time = time.time() - start_time
            
            # 保存去雾结果
            save_path = os.path.join(save_dir, filename)
            torchvision.utils.save_image(dehaze_image, save_path)
            
            # 打印处理信息和指标
            print(f"[{i+1}/{len(file_list)}] 处理: {filename} (耗时: {process_time:.2f}秒)")
            
            if metrics:
                print(f"  - PSNR: {metrics['psnr']:.4f}dB, SSIM: {metrics['ssim']:.4f}, MSE: {metrics['mse']:.6f}")
                # 累加指标
                for key in total_metrics:
                    total_metrics[key] += metrics[key]
                processed_count += 1
    
    # 打印平均指标
    if processed_count > 0:
        print("\n图像质量评估结果:")
        print(f"平均 PSNR: {total_metrics['psnr'] / processed_count:.4f} dB")
        print(f"平均 SSIM: {total_metrics['ssim'] / processed_count:.4f}")
        print(f"平均 MSE: {total_metrics['mse'] / processed_count:.6f}")
    
    print(f"\n所有图像处理完成，结果已保存到: {save_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-td", "--test_directory", required=True, help="测试图像目录路径")
    ap.add_argument("-gd", "--gt_directory", help="Ground Truth图像目录路径(可选)")
    args = vars(ap.parse_args())
    multiple_dehaze_test(args["test_directory"], args.get("gt_directory"))