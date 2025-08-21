# -*- coding: utf-8 -*-
"""
克拉尼图形数据预处理模块
专门用于克拉尼图形的数据加载、预处理和特征提取
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter
import json
from datetime import datetime

from .sift_extractor import ChladniSIFTExtractor

class ChladniDataPreprocessor:
    """
    克拉尼图形数据预处理器
    整合图像预处理、SIFT特征提取和数据管理功能
    """
    
    def __init__(self, image_size=(256, 256), sift_params=None):
        """
        初始化预处理器
        
        Args:
            image_size: 图像统一尺寸 (width, height)
            sift_params: SIFT参数字典
        """
        self.image_size = image_size
        
        # 初始化SIFT特征提取器
        if sift_params is None:
            sift_params = {
                'step_size': 8,
                'patch_size': 16,
                'vocab_size': 100
            }
        
        self.sift_extractor = ChladniSIFTExtractor(**sift_params)
        
        # 数据存储
        self.images = []
        self.labels = []
        self.image_paths = []
        self.class_names = []
        self.features = None
        
        # 统计信息
        self.stats = {
            'total_images': 0,
            'classes': {},
            'image_sizes': [],
            'load_time': None
        }
    
    def load_images_from_directory(self, data_dir, supported_formats=None):
        """
        从目录结构加载图像数据
        
        目录结构应为:
        data_dir/
        ├── class1/
        │   ├── image1.jpg
        │   └── image2.jpg
        └── class2/
            ├── image3.jpg
            └── image4.jpg
        
        Args:
            data_dir: 数据根目录
            supported_formats: 支持的图像格式列表
        """
        if supported_formats is None:
            supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        print(f"开始从 {data_dir} 加载克拉尼图形数据...")
        start_time = datetime.now()
        
        if not os.path.exists(data_dir):
            raise ValueError(f"数据目录不存在: {data_dir}")
        
        # 清空之前的数据
        self.images = []
        self.labels = []
        self.image_paths = []
        self.class_names = []
        
        # 获取类别目录
        class_dirs = [d for d in os.listdir(data_dir) 
                     if os.path.isdir(os.path.join(data_dir, d))]
        
        if not class_dirs:
            raise ValueError(f"在 {data_dir} 中未找到类别子目录")
        
        self.class_names = sorted(class_dirs)
        print(f"发现 {len(self.class_names)} 个类别: {self.class_names}")
        
        # 加载每个类别的图像
        for class_name in self.class_names:
            class_dir = os.path.join(data_dir, class_name)
            class_images = []
            
            # 获取该类别的所有图像文件
            for filename in os.listdir(class_dir):
                if any(filename.lower().endswith(fmt) for fmt in supported_formats):
                    image_path = os.path.join(class_dir, filename)
                    
                    # 加载和预处理图像
                    image = self._load_and_preprocess_image(image_path)
                    
                    if image is not None:
                        self.images.append(image)
                        self.labels.append(class_name)
                        self.image_paths.append(image_path)
                        class_images.append(image)
            
            print(f"  {class_name}: 加载了 {len(class_images)} 张图像")
        
        # 更新统计信息
        self._update_stats()
        
        load_time = (datetime.now() - start_time).total_seconds()
        self.stats['load_time'] = load_time
        
        print(f"\n数据加载完成！")
        print(f"总计: {len(self.images)} 张图像，{len(self.class_names)} 个类别")
        print(f"加载耗时: {load_time:.2f} 秒")
        
        return len(self.images)
    
    def _load_and_preprocess_image(self, image_path):
        """
        加载并预处理单张图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            processed_image: 预处理后的图像
        """
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                print(f"警告: 无法读取图像 {image_path}")
                return None
            
            # 转换为RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 记录原始尺寸
            original_size = image.shape[:2]
            self.stats['image_sizes'].append(original_size)
            
            # 调整尺寸
            if image.shape[:2] != self.image_size:
                image = cv2.resize(image, self.image_size)
            
            # 克拉尼图形特殊预处理
            image = self._chladni_specific_preprocessing(image)
            
            return image
            
        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {str(e)}")
            return None
    
    def _chladni_specific_preprocessing(self, image):
        """
        克拉尼图形特殊预处理
        
        Args:
            image: 输入图像
            
        Returns:
            processed_image: 处理后的图像
        """
        # 转换为灰度图（克拉尼图形通常是单色的）
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # 直方图均衡化增强对比度
        enhanced = cv2.equalizeHist(gray)
        
        # 高斯滤波去噪
        denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # 可选：边缘增强
        # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        # sharpened = cv2.filter2D(denoised, -1, kernel)
        
        return denoised
    
    def extract_sift_features(self, build_vocabulary=True):
        """
        提取所有图像的SIFT特征
        
        Args:
            build_vocabulary: 是否构建词汇表
            
        Returns:
            features: 特征矩阵
        """
        if not self.images:
            raise ValueError("请先加载图像数据")
        
        print(f"开始提取 {len(self.images)} 张图像的SIFT特征...")
        
        # 构建词汇表
        if build_vocabulary:
            print("构建SIFT词汇表...")
            self.sift_extractor.build_vocabulary(self.images)
        
        # 提取特征
        print("提取BoW特征...")
        self.features = self.sift_extractor.extract_bow_features_batch(self.images)
        
        print(f"特征提取完成！特征维度: {self.features.shape}")
        
        return self.features
    
    def split_dataset(self, test_size=0.2, val_size=0.1, random_state=42, stratify=True):
        """
        分割数据集
        
        Args:
            test_size: 测试集比例
            val_size: 验证集比例
            random_state: 随机种子
            stratify: 是否分层采样
            
        Returns:
            split_data: 分割后的数据字典
        """
        if self.features is None:
            raise ValueError("请先提取特征")
        
        X = self.features
        y = np.array(self.labels)
        
        # 分层采样参数
        stratify_param = y if stratify else None
        
        # 首先分出测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=stratify_param
        )
        
        # 从剩余数据中分出验证集
        if val_size > 0:
            val_size_adjusted = val_size / (1 - test_size)  # 调整验证集比例
            stratify_temp = y_temp if stratify else None
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted, 
                random_state=random_state, stratify=stratify_temp
            )
        else:
            X_train, y_train = X_temp, y_temp
            X_val, y_val = None, None
        
        # 创建分割结果
        split_data = {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'X_val': X_val,
            'y_val': y_val
        }
        
        # 打印分割信息
        print(f"\n=== 数据集分割结果 ===")
        print(f"训练集: {len(X_train)} 样本")
        if X_val is not None:
            print(f"验证集: {len(X_val)} 样本")
        print(f"测试集: {len(X_test)} 样本")
        
        # 打印各类别分布
        print("\n各集合类别分布:")
        for split_name, split_labels in [('训练集', y_train), ('测试集', y_test)]:
            if split_labels is not None:
                class_counts = Counter(split_labels)
                print(f"  {split_name}:")
                for class_name in self.class_names:
                    count = class_counts.get(class_name, 0)
                    percentage = (count / len(split_labels)) * 100
                    print(f"    {class_name}: {count} ({percentage:.1f}%)")
        
        if y_val is not None:
            class_counts = Counter(y_val)
            print(f"  验证集:")
            for class_name in self.class_names:
                count = class_counts.get(class_name, 0)
                percentage = (count / len(y_val)) * 100
                print(f"    {class_name}: {count} ({percentage:.1f}%)")
        
        return split_data
    
    def _update_stats(self):
        """
        更新统计信息
        """
        self.stats['total_images'] = len(self.images)
        
        # 类别统计
        class_counts = Counter(self.labels)
        self.stats['classes'] = dict(class_counts)
    
    def get_dataset_info(self):
        """
        获取数据集信息
        
        Returns:
            info: 数据集信息字典
        """
        info = {
            'total_images': len(self.images),
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'class_distribution': dict(Counter(self.labels)),
            'image_size': self.image_size,
            'feature_dimension': self.features.shape[1] if self.features is not None else None,
            'sift_params': {
                'step_size': self.sift_extractor.step_size,
                'patch_size': self.sift_extractor.patch_size,
                'vocab_size': self.sift_extractor.vocab_size
            }
        }
        
        return info
    
    def visualize_samples(self, samples_per_class=3, figsize=(15, 10)):
        """
        可视化每个类别的样本图像
        
        Args:
            samples_per_class: 每个类别显示的样本数
            figsize: 图像大小
        """
        if not self.images:
            print("没有加载的图像数据")
            return
        
        num_classes = len(self.class_names)
        fig, axes = plt.subplots(num_classes, samples_per_class, 
                                figsize=figsize)
        
        if num_classes == 1:
            axes = axes.reshape(1, -1)
        
        for i, class_name in enumerate(self.class_names):
            # 获取该类别的图像索引
            class_indices = [j for j, label in enumerate(self.labels) 
                           if label == class_name]
            
            # 随机选择样本
            selected_indices = np.random.choice(
                class_indices, 
                min(samples_per_class, len(class_indices)), 
                replace=False
            )
            
            for j in range(samples_per_class):
                ax = axes[i, j] if num_classes > 1 else axes[j]
                
                if j < len(selected_indices):
                    img_idx = selected_indices[j]
                    image = self.images[img_idx]
                    
                    # 显示图像
                    if len(image.shape) == 2:  # 灰度图
                        ax.imshow(image, cmap='gray')
                    else:  # 彩色图
                        ax.imshow(image)
                    
                    ax.set_title(f"{class_name}\n{os.path.basename(self.image_paths[img_idx])}")
                else:
                    ax.axis('off')
                
                ax.set_xticks([])
                ax.set_yticks([])
        
        plt.tight_layout()
        plt.suptitle('克拉尼图形样本展示', fontsize=16, y=1.02)
        plt.show()
    
    def save_preprocessed_data(self, save_dir):
        """
        保存预处理后的数据
        
        Args:
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存特征和标签
        if self.features is not None:
            np.save(os.path.join(save_dir, 'features.npy'), self.features)
        np.save(os.path.join(save_dir, 'labels.npy'), np.array(self.labels))
        
        # 保存元数据
        metadata = {
            'class_names': self.class_names,
            'image_paths': self.image_paths,
            'image_size': self.image_size,
            'stats': self.stats,
            'sift_params': {
                'step_size': self.sift_extractor.step_size,
                'patch_size': self.sift_extractor.patch_size,
                'vocab_size': self.sift_extractor.vocab_size
            }
        }
        
        with open(os.path.join(save_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # 保存SIFT词汇表
        if hasattr(self.sift_extractor, 'kmeans') and self.sift_extractor.kmeans is not None:
            self.sift_extractor.save_vocabulary(os.path.join(save_dir, 'sift_vocabulary.pkl'))
        
        print(f"预处理数据已保存到: {save_dir}")
    
    def load_preprocessed_data(self, save_dir):
        """
        加载预处理后的数据
        
        Args:
            save_dir: 数据目录
        """
        # 加载特征和标签
        features_path = os.path.join(save_dir, 'features.npy')
        if os.path.exists(features_path):
            self.features = np.load(features_path)
        
        labels_path = os.path.join(save_dir, 'labels.npy')
        if os.path.exists(labels_path):
            self.labels = np.load(labels_path).tolist()
        
        # 加载元数据
        metadata_path = os.path.join(save_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            self.class_names = metadata.get('class_names', [])
            self.image_paths = metadata.get('image_paths', [])
            self.image_size = tuple(metadata.get('image_size', (256, 256)))
            self.stats = metadata.get('stats', {})
        
        # 加载SIFT词汇表
        vocab_path = os.path.join(save_dir, 'sift_vocabulary.pkl')
        if os.path.exists(vocab_path):
            self.sift_extractor.load_vocabulary(vocab_path)
        
        print(f"预处理数据已从 {save_dir} 加载")
        print(f"特征维度: {self.features.shape if self.features is not None else 'None'}")
        print(f"类别数量: {len(self.class_names)}")


if __name__ == "__main__":
    # 测试代码
    print("克拉尼图形数据预处理器测试")
    
    # 创建预处理器
    preprocessor = ChladniDataPreprocessor()
    
    # 这里需要实际的数据目录进行测试
    # preprocessor.load_images_from_directory('path/to/chladni/data')
    # preprocessor.extract_sift_features()
    # split_data = preprocessor.split_dataset()
    
    print("预处理器创建成功！")