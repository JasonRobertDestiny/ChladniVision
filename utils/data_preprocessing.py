# -*- coding: utf-8 -*-
"""
图像分类数据预处理模块
包含图像标准化、增强、数据分割等功能
"""

import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from PIL import Image
import matplotlib.pyplot as plt

class ImagePreprocessor:
    """
    图像预处理类
    """
    
    def __init__(self, target_size=(224, 224)):
        """
        初始化预处理器
        
        Args:
            target_size: 目标图像尺寸 (height, width)
        """
        self.target_size = target_size
        
    def normalize_image(self, image):
        """
        图像标准化：将像素值缩放到0-1范围
        
        Args:
            image: 输入图像数组
            
        Returns:
            normalized_image: 标准化后的图像
        """
        if isinstance(image, str):
            # 如果输入是文件路径
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 确保图像是float32类型
        if image.dtype != np.float32:
            image = image.astype(np.float32)
            
        # 标准化到0-1范围
        normalized_image = image / 255.0
        return normalized_image
    
    def resize_image(self, image, target_size=None):
        """
        调整图像尺寸
        
        Args:
            image: 输入图像
            target_size: 目标尺寸，如果为None则使用初始化时的尺寸
            
        Returns:
            resized_image: 调整尺寸后的图像
        """
        if target_size is None:
            target_size = self.target_size
            
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        resized_image = cv2.resize(image, target_size)
        return resized_image
    
    def load_and_preprocess_image(self, image_path):
        """
        加载并预处理单张图像
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            processed_image: 预处理后的图像
        """
        # 加载图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")
            
        # 转换颜色空间
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 调整尺寸
        image = self.resize_image(image)
        
        # 标准化
        image = self.normalize_image(image)
        
        return image
    
    def create_data_generator(self, 
                            rotation_range=40,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            shear_range=0.2,
                            zoom_range=0.2,
                            horizontal_flip=True,
                            fill_mode='nearest',
                            validation_split=0.2):
        """
        创建数据增强生成器
        
        Args:
            rotation_range: 旋转角度范围
            width_shift_range: 宽度偏移范围
            height_shift_range: 高度偏移范围
            shear_range: 剪切变换范围
            zoom_range: 缩放范围
            horizontal_flip: 是否水平翻转
            fill_mode: 填充模式
            validation_split: 验证集比例
            
        Returns:
            datagen: 数据生成器
        """
        datagen = ImageDataGenerator(
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            horizontal_flip=horizontal_flip,
            fill_mode=fill_mode,
            rescale=1./255,  # 标准化
            validation_split=validation_split
        )
        
        return datagen
    
    def load_dataset_from_directory(self, data_dir):
        """
        从目录结构加载数据集
        假设目录结构为：
        data_dir/
        ├── class1/
        │   ├── img1.jpg
        │   └── img2.jpg
        └── class2/
            ├── img3.jpg
            └── img4.jpg
            
        Args:
            data_dir: 数据目录路径
            
        Returns:
            images: 图像数组
            labels: 标签数组
            class_names: 类别名称列表
        """
        images = []
        labels = []
        class_names = []
        
        # 获取所有类别目录
        for class_idx, class_name in enumerate(sorted(os.listdir(data_dir))):
            class_path = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_path):
                continue
                
            class_names.append(class_name)
            
            # 加载该类别下的所有图像
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(class_path, img_name)
                    try:
                        img = self.load_and_preprocess_image(img_path)
                        images.append(img)
                        labels.append(class_idx)
                    except Exception as e:
                        print(f"加载图像失败 {img_path}: {e}")
        
        images = np.array(images)
        labels = np.array(labels)
        
        return images, labels, class_names
    
    def split_dataset(self, images, labels, test_size=0.2, val_size=0.1, random_state=42):
        """
        分割数据集为训练集、验证集和测试集
        
        Args:
            images: 图像数组
            labels: 标签数组
            test_size: 测试集比例
            val_size: 验证集比例（相对于训练集）
            random_state: 随机种子
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # 首先分离出测试集
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            images, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        # 再从训练集中分离出验证集
        if val_size > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=val_size, 
                random_state=random_state, stratify=y_train_val
            )
        else:
            X_train, X_val, y_train, y_val = X_train_val, None, y_train_val, None
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def prepare_labels(self, labels, num_classes=None):
        """
        准备标签数据（one-hot编码）
        
        Args:
            labels: 原始标签
            num_classes: 类别数量，如果为None则自动计算
            
        Returns:
            encoded_labels: one-hot编码后的标签
        """
        if num_classes is None:
            num_classes = len(np.unique(labels))
            
        encoded_labels = to_categorical(labels, num_classes)
        return encoded_labels
    
    def visualize_samples(self, images, labels, class_names, num_samples=9):
        """
        可视化数据样本
        
        Args:
            images: 图像数组
            labels: 标签数组
            class_names: 类别名称列表
            num_samples: 显示的样本数量
        """
        plt.figure(figsize=(12, 8))
        
        for i in range(min(num_samples, len(images))):
            plt.subplot(3, 3, i + 1)
            
            # 显示图像
            img = images[i]
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            
            plt.imshow(img)
            plt.title(f'类别: {class_names[labels[i]]}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()


def create_sample_dataset(output_dir='data/sample'):
    """
    创建示例数据集（用于演示）
    
    Args:
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建两个类别的示例目录
    for class_name in ['cats', 'dogs']:
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
    
    print(f"示例数据集目录已创建: {output_dir}")
    print("请将图像文件放入相应的类别文件夹中：")
    print(f"- 猫的图像放入: {output_dir}/cats/")
    print(f"- 狗的图像放入: {output_dir}/dogs/")


if __name__ == "__main__":
    # 创建示例数据集目录
    create_sample_dataset()
    
    # 演示预处理功能
    preprocessor = ImagePreprocessor(target_size=(224, 224))
    
    print("数据预处理模块已准备就绪！")
    print("主要功能：")
    print("1. 图像标准化和尺寸调整")
    print("2. 数据增强")
    print("3. 数据集加载和分割")
    print("4. 标签编码")
    print("5. 数据可视化")