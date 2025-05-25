#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MNIST模型图像识别工具
功能：
1. 加载./models文件夹下指定的模型
2. 识别./data/test/png文件夹下的所有图片或指定图片
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os
import glob
import argparse
import json
from datetime import datetime


# --- 模型定义（与训练时保持一致）---
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout层
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        # 全连接层
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        # 第一个卷积块
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        # 第二个卷积块
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # 第三个卷积块
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # 展平
        x = x.view(-1, 128 * 3 * 3)
        x = self.dropout1(x)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class MNISTRecognizer:
    """MNIST数字识别器"""

    def __init__(self, model_path=None):
        """
        初始化识别器

        参数:
            model_path (str): 模型文件路径，如果为None则使用最佳模型
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 使用设备: {self.device}")

        # 图像预处理变换（与训练时保持一致）
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # 确保是灰度图
            transforms.Resize((28, 28)),  # 调整到28x28
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
        ])

        # 加载模型
        self.model = self.load_model(model_path)

    def load_model(self, model_path=None):
        """
        加载训练好的模型

        参数:
            model_path (str): 模型文件路径

        返回:
            torch.nn.Module: 加载的模型
        """
        if model_path is None:
            # 自动选择最佳模型
            model_path = self.find_best_model()

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        print(f"📦 正在加载模型: {model_path}")

        # 创建模型实例
        model = CNN()

        # 加载模型参数
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint)
            model.to(self.device)
            model.eval()  # 设置为评估模式
            print("✅ 模型加载成功！")
            return model
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {e}")

    def find_best_model(self):
        """
        自动查找最佳模型文件

        返回:
            str: 最佳模型的路径
        """
        models_dir = "./models"

        # 优先选择best模型
        best_model = os.path.join(models_dir, "best_mnist_cnn_model.pth")
        if os.path.exists(best_model):
            return best_model

        # 如果没有best模型，选择最新的epoch模型
        pattern = os.path.join(models_dir, "mnist_cnn_epoch_*.pth")
        model_files = glob.glob(pattern)

        if not model_files:
            raise FileNotFoundError("在./models目录下没有找到任何模型文件")

        # 按epoch数字排序，选择最大的
        def extract_epoch(filename):
            try:
                epoch_str = filename.split("epoch_")[1].split(".pth")[0]
                return int(epoch_str)
            except:
                return 0

        latest_model = max(model_files, key=extract_epoch)
        return latest_model

    def list_available_models(self):
        """
        列出所有可用的模型文件

        返回:
            list: 可用模型文件列表
        """
        models_dir = "./models"
        if not os.path.exists(models_dir):
            return []

        model_files = glob.glob(os.path.join(models_dir, "*.pth"))
        return sorted(model_files)

    def select_model_interactive(self):
        """
        交互式选择模型

        返回:
            str: 选择的模型路径
        """
        model_files = self.list_available_models()

        if not model_files:
            print("❌ 没有找到可用的模型文件")
            return None

        print("\n📁 可用模型列表:")
        print("-" * 50)

        # 显示模型列表
        for i, model_file in enumerate(model_files, 1):
            filename = os.path.basename(model_file)
            file_size = os.path.getsize(model_file) / (1024*1024)  # MB

            # 获取创建时间
            create_time = os.path.getctime(model_file)
            create_time_str = datetime.fromtimestamp(
                create_time).strftime('%Y-%m-%d %H:%M:%S')

            # 标记最佳模型
            is_best = "best" in filename.lower()
            mark = " ⭐ [推荐]" if is_best else ""

            print(
                f"   {i:2d}. {filename} ({file_size:.2f} MB) - {create_time_str}{mark}")

        print("-" * 50)

        while True:
            try:
                choice = input("请选择模型编号 (输入数字，0自动选择最佳模型): ").strip()

                if choice == '0':
                    # 自动选择最佳模型
                    return self.find_best_model()

                choice_num = int(choice)
                if 1 <= choice_num <= len(model_files):
                    selected_model = model_files[choice_num - 1]
                    print(f"✅ 已选择模型: {os.path.basename(selected_model)}")
                    return selected_model
                else:
                    print(f"❌ 请输入 0-{len(model_files)} 之间的数字")

            except ValueError:
                print("❌ 请输入有效的数字")
            except KeyboardInterrupt:
                print("\n⚠️  操作被取消")
                return None

    def predict_single_image(self, image_path):
        """
        识别单张图片

        参数:
            image_path (str): 图片路径

        返回:
            tuple: (预测类别, 置信度, 所有类别的概率)
        """
        try:
            # 加载并预处理图片
            image = Image.open(image_path)

            # 如果是RGBA模式，转换为RGB
            if image.mode == 'RGBA':
                # 创建白色背景
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background

            # 应用变换
            image_tensor = self.transform(image).unsqueeze(0)  # 添加batch维度
            image_tensor = image_tensor.to(self.device)

            # 进行预测
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                predicted_class = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0][predicted_class].item()

            return predicted_class, confidence, probabilities[0].cpu().numpy()

        except Exception as e:
            print(f"❌ 处理图片 {image_path} 时出错: {e}")
            return None, None, None

    def predict_multiple_images(self, images_dir, pattern="*.png"):
        """
        批量识别图片

        参数:
            images_dir (str): 图片目录路径
            pattern (str): 文件匹配模式

        返回:
            list: 识别结果列表
        """
        # 获取所有匹配的图片文件
        search_pattern = os.path.join(images_dir, pattern)
        image_files = glob.glob(search_pattern)

        if not image_files:
            print(f"⚠️  在目录 {images_dir} 中没有找到匹配的图片文件")
            return []

        print(f"📁 找到 {len(image_files)} 个图片文件")
        print("=" * 60)

        results = []
        for i, img_path in enumerate(image_files, 1):
            filename = os.path.basename(img_path)
            print(f"[{i}/{len(image_files)}] 正在识别: {filename}")

            predicted_class, confidence, probabilities = self.predict_single_image(
                img_path)

            if predicted_class is not None:
                result = {
                    'filename': filename,
                    'path': img_path,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'probabilities': probabilities
                }
                results.append(result)

                print(f"   🎯 预测结果: {predicted_class}")
                print(f"   📊 置信度: {confidence:.4f} ({confidence*100:.2f}%)")

                # 显示真实标签（如果文件名包含标签信息）
                true_label = self.extract_true_label(filename)
                if true_label is not None:
                    is_correct = predicted_class == true_label
                    status = "✅ 正确" if is_correct else "❌ 错误"
                    print(f"   🏷️  真实标签: {true_label} - {status}")

                print("-" * 40)

        return results

    def extract_true_label(self, filename):

        # 从文件名中提取真实标签
        # 参数: filename(str): 文件名
        # 返回: int or None: 真实标签
        try:
            # 假设文件名格式为 number3_2.png，提取数字2（下划线后的数字）
            if filename.startswith('number') and '_' in filename:
                # 分割文件名，获取下划线后的部分
                parts = filename.split('_')
                if len(parts) >= 2:
                    # 获取下划线后的数字部分，去掉.png后缀
                    label_str = parts[1].split('.')[0]
                    return int(label_str)
        except:
            pass
        return None

    def print_summary(self, results):
        """
        打印识别结果摘要

        参数:
            results (list): 识别结果列表
        """
        if not results:
            print("📝 没有识别结果")
            return

        print("\n" + "=" * 60)
        print("                   📊 识别结果摘要")
        print("=" * 60)

        total_images = len(results)
        correct_predictions = 0

        # 统计准确率
        for result in results:
            filename = result['filename']
            predicted = result['predicted_class']
            true_label = self.extract_true_label(filename)

            if true_label is not None and predicted == true_label:
                correct_predictions += 1

        print(f"📁 总图片数量: {total_images}")

        if correct_predictions > 0:
            accuracy = correct_predictions / total_images * 100
            print(
                f"🎯 识别准确率: {correct_predictions}/{total_images} ({accuracy:.2f}%)")

        # 统计各类别的识别情况
        class_counts = {}
        for result in results:
            predicted = result['predicted_class']
            class_counts[predicted] = class_counts.get(predicted, 0) + 1

        print(f"\n📊 各数字识别统计:")
        for digit in sorted(class_counts.keys()):
            count = class_counts[digit]
            percentage = count / total_images * 100
            print(f"   数字 {digit}: {count} 次 ({percentage:.1f}%)")

        # 显示最高和最低置信度
        confidences = [r['confidence'] for r in results]
        if confidences:
            max_conf = max(confidences)
            min_conf = min(confidences)
            avg_conf = sum(confidences) / len(confidences)

            print(f"\n📈 置信度统计:")
            print(f"   最高置信度: {max_conf:.4f} ({max_conf*100:.2f}%)")
            print(f"   最低置信度: {min_conf:.4f} ({min_conf*100:.2f}%)")
            print(f"   平均置信度: {avg_conf:.4f} ({avg_conf*100:.2f}%)")


def interactive_mode():
    """交互模式"""
    print("=" * 60)
    print("          🔍 MNIST数字识别工具 - 交互模式")
    print("=" * 60)
    print("功能：使用训练好的CNN模型识别手写数字图片")
    print("支持的命令：")
    print("  - 输入图片路径：识别单张图片")
    print("  - 'batch'：批量识别./data/test/png/目录下所有图片")
    print("  - 'list'：显示可用的模型列表")
    print("  - 'select'：选择不同的模型")
    print("  - 'model'：查看当前使用的模型")
    print("  - 'quit' 或 'exit'：退出程序")
    print("-" * 60)

    # 初始化识别器
    try:
        # 先显示模型选择界面
        print("\n🎯 请选择要使用的模型:")
        recognizer = MNISTRecognizer()
        model_path = recognizer.select_model_interactive()

        if model_path is None:
            print("❌ 未选择模型，程序退出")
            return

        # 重新初始化识别器使用选择的模型
        recognizer = MNISTRecognizer(model_path)
        current_model = os.path.basename(model_path)

    except Exception as e:
        print(f"❌ 初始化识别器失败: {e}")
        return

    while True:
        try:
            user_input = input(
                f"\n🎯 请输入命令或图片路径 [当前模型: {current_model}]: ").strip()

            # 检查退出命令
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 感谢使用！再见！")
                break

            # 检查输入是否为空
            if not user_input:
                print("⚠️  请输入有效的命令或图片路径")
                continue

            # 选择模型命令
            if user_input.lower() == 'select':
                print("\n🔄 重新选择模型...")
                new_model_path = recognizer.select_model_interactive()
                if new_model_path and new_model_path != model_path:
                    try:
                        recognizer = MNISTRecognizer(new_model_path)
                        model_path = new_model_path
                        current_model = os.path.basename(model_path)
                        print(f"✅ 模型切换成功: {current_model}")
                    except Exception as e:
                        print(f"❌ 模型切换失败: {e}")
                continue

            # 查看当前模型命令
            if user_input.lower() == 'model':
                print(f"\n📦 当前使用模型: {current_model}")
                print(f"📍 模型路径: {model_path}")
                file_size = os.path.getsize(model_path) / (1024*1024)
                create_time = os.path.getctime(model_path)
                create_time_str = datetime.fromtimestamp(
                    create_time).strftime('%Y-%m-%d %H:%M:%S')
                print(f"📏 文件大小: {file_size:.2f} MB")
                print(f"🕒 创建时间: {create_time_str}")
                continue

            # 批量处理命令
            if user_input.lower() == 'batch':
                print("\n🔄 开始批量识别...")
                results = recognizer.predict_multiple_images(
                    "./data/test/png/")
                recognizer.print_summary(results)
                continue

            # 显示模型列表
            if user_input.lower() == 'list':
                models_dir = "./models"
                if os.path.exists(models_dir):
                    model_files = glob.glob(os.path.join(models_dir, "*.pth"))
                    print(f"\n📁 可用模型列表 ({models_dir}):")
                    for model_file in sorted(model_files):
                        filename = os.path.basename(model_file)
                        file_size = os.path.getsize(
                            model_file) / (1024*1024)  # MB
                        is_current = filename == current_model
                        mark = " ✅ [当前]" if is_current else ""
                        print(f"   📦 {filename} ({file_size:.2f} MB){mark}")
                else:
                    print("❌ models目录不存在")
                continue

            # 识别单张图片
            if os.path.exists(user_input):
                print(f"\n🔍 正在识别图片: {user_input}")
                print("-" * 40)

                predicted_class, confidence, probabilities = recognizer.predict_single_image(
                    user_input)

                if predicted_class is not None:
                    print(f"🎯 预测结果: {predicted_class}")
                    print(f"📊 置信度: {confidence:.4f} ({confidence*100:.2f}%)")

                    # 显示所有类别的概率分布
                    print("\n📈 各数字的概率分布:")
                    for i, prob in enumerate(probabilities):
                        bar_length = int(prob * 20)  # 简单的进度条
                        bar = "█" * bar_length + "░" * (20 - bar_length)
                        print(f"   数字 {i}: {bar} {prob:.4f} ({prob*100:.2f}%)")

                    # 检查真实标签
                    filename = os.path.basename(user_input)
                    true_label = recognizer.extract_true_label(filename)
                    if true_label is not None:
                        is_correct = predicted_class == true_label
                        status = "✅ 正确" if is_correct else "❌ 错误"
                        print(f"\n🏷️  真实标签: {true_label} - {status}")
                else:
                    print("❌ 识别失败")
            else:
                print(f"❌ 文件不存在: {user_input}")

        except KeyboardInterrupt:
            print("\n\n👋 程序被用户中断，正在退出...")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='MNIST数字识别工具')
    parser.add_argument('-m', '--model', help='指定模型文件路径')
    parser.add_argument('-i', '--image', help='单张图片路径')
    parser.add_argument('-d', '--directory',
                        default='./data/test/png/', help='图片目录路径')
    parser.add_argument('-b', '--batch', action='store_true', help='批量处理模式')
    parser.add_argument('-p', '--pattern', default='*.png', help='文件匹配模式')
    parser.add_argument('-s', '--select', action='store_true', help='交互式选择模型')

    args = parser.parse_args()

    try:
        # 如果指定了交互式选择模型
        if args.select:
            temp_recognizer = MNISTRecognizer()
            selected_model = temp_recognizer.select_model_interactive()
            if selected_model:
                args.model = selected_model
            else:
                print("❌ 未选择模型，程序退出")
                return

        # 初始化识别器
        recognizer = MNISTRecognizer(args.model)

        if args.image:
            # 单张图片识别
            print(f"🔍 识别单张图片: {args.image}")
            predicted_class, confidence, probabilities = recognizer.predict_single_image(
                args.image)

            if predicted_class is not None:
                print(f"🎯 预测结果: {predicted_class}")
                print(f"📊 置信度: {confidence:.4f} ({confidence*100:.2f}%)")

                # 显示真实标签
                filename = os.path.basename(args.image)
                true_label = recognizer.extract_true_label(filename)
                if true_label is not None:
                    is_correct = predicted_class == true_label
                    status = "✅ 正确" if is_correct else "❌ 错误"
                    print(f"🏷️  真实标签: {true_label} - {status}")
            else:
                print("❌ 识别失败")

        elif args.batch:
            # 批量识别
            print(f"🔄 批量识别目录: {args.directory}")
            results = recognizer.predict_multiple_images(
                args.directory, args.pattern)
            recognizer.print_summary(results)

        else:
            # 交互模式
            interactive_mode()

    except Exception as e:
        print(f"❌ 程序执行失败: {e}")


if __name__ == "__main__":
    # 检查是否有命令行参数
    import sys

    if len(sys.argv) > 1:
        # 有命令行参数，使用命令行模式
        main()
    else:
        # 没有命令行参数，进入交互模式
        interactive_mode()
