# -*- coding: utf-8 -*-
"""
克拉尼图形SIFT特征提取模块
实现Dense SIFT特征提取，用于图像分类
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle
import os

class DenseSIFTExtractor:
    """
    Dense SIFT特征提取器
    用于提取图像的密集SIFT特征，适用于克拉尼图形等纹理丰富的图像
    """
    
    def __init__(self, step_size=10, patch_size=16, vocab_size=100):
        """
        初始化Dense SIFT提取器
        
        Args:
            step_size: 密集采样的步长
            patch_size: SIFT特征的patch大小
            vocab_size: 词汇表大小（用于Bag of Words）
        """
        self.step_size = step_size
        self.patch_size = patch_size
        self.vocab_size = vocab_size
        
        # 创建SIFT检测器
        self.sift = cv2.SIFT_create()
        
        # 用于Bag of Words的聚类器和标准化器
        self.kmeans = None
        self.scaler = StandardScaler()
        self.vocabulary = None
        
    def extract_dense_sift(self, image):
        """
        提取图像的Dense SIFT特征
        
        Args:
            image: 输入图像（灰度图或彩色图）
            
        Returns:
            descriptors: SIFT描述符数组
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
            
        # 确保图像是uint8类型
        if gray.dtype != np.uint8:
            gray = (gray * 255).astype(np.uint8)
        
        # 生成密集关键点
        keypoints = []
        h, w = gray.shape
        
        for y in range(0, h - self.patch_size, self.step_size):
            for x in range(0, w - self.patch_size, self.step_size):
                keypoint = cv2.KeyPoint(x + self.patch_size//2, 
                                      y + self.patch_size//2, 
                                      self.patch_size)
                keypoints.append(keypoint)
        
        # 计算SIFT描述符
        keypoints, descriptors = self.sift.compute(gray, keypoints)
        
        if descriptors is None:
            return np.array([])
            
        return descriptors
    
    def build_vocabulary(self, images):
        """
        构建SIFT特征的词汇表（Bag of Words）
        
        Args:
            images: 训练图像列表
        """
        print(f"正在构建SIFT词汇表，词汇表大小: {self.vocab_size}")
        
        all_descriptors = []
        
        # 提取所有图像的SIFT特征
        for i, image in enumerate(images):
            if i % 10 == 0:
                print(f"处理图像 {i+1}/{len(images)}")
                
            descriptors = self.extract_dense_sift(image)
            if len(descriptors) > 0:
                all_descriptors.append(descriptors)
        
        if not all_descriptors:
            raise ValueError("无法从图像中提取SIFT特征")
            
        # 合并所有描述符
        all_descriptors = np.vstack(all_descriptors)
        print(f"总共提取了 {len(all_descriptors)} 个SIFT描述符")
        
        # 使用K-means聚类构建词汇表
        print("正在进行K-means聚类...")
        self.kmeans = KMeans(n_clusters=self.vocab_size, random_state=42, n_init=10)
        self.kmeans.fit(all_descriptors)
        
        self.vocabulary = self.kmeans.cluster_centers_
        print("词汇表构建完成")
    
    def extract_bow_features(self, image):
        """
        提取图像的Bag of Words特征
        
        Args:
            image: 输入图像
            
        Returns:
            bow_features: Bag of Words特征向量
        """
        if self.kmeans is None:
            raise ValueError("请先调用build_vocabulary()构建词汇表")
            
        # 提取SIFT描述符
        descriptors = self.extract_dense_sift(image)
        
        if len(descriptors) == 0:
            return np.zeros(self.vocab_size)
        
        # 将描述符分配到最近的聚类中心
        labels = self.kmeans.predict(descriptors)
        
        # 构建直方图
        bow_features = np.bincount(labels, minlength=self.vocab_size)
        
        # 归一化
        if bow_features.sum() > 0:
            bow_features = bow_features.astype(float) / bow_features.sum()
            
        return bow_features
    
    def extract_features_batch(self, images):
        """
        批量提取图像特征
        
        Args:
            images: 图像列表
            
        Returns:
            features: 特征矩阵
        """
        features = []
        
        for i, image in enumerate(images):
            if i % 10 == 0:
                print(f"提取特征 {i+1}/{len(images)}")
                
            bow_features = self.extract_bow_features(image)
            features.append(bow_features)
        
        features = np.array(features)
        
        # 标准化特征
        if not hasattr(self.scaler, 'mean_'):
            features = self.scaler.fit_transform(features)
        else:
            features = self.scaler.transform(features)
            
        return features
    
    def save_model(self, filepath):
        """
        保存特征提取器模型
        
        Args:
            filepath: 保存路径
        """
        model_data = {
            'step_size': self.step_size,
            'patch_size': self.patch_size,
            'vocab_size': self.vocab_size,
            'kmeans': self.kmeans,
            'scaler': self.scaler,
            'vocabulary': self.vocabulary
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"SIFT特征提取器已保存到: {filepath}")
    
    def load_model(self, filepath):
        """
        加载特征提取器模型
        
        Args:
            filepath: 模型文件路径
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.step_size = model_data['step_size']
        self.patch_size = model_data['patch_size']
        self.vocab_size = model_data['vocab_size']
        self.kmeans = model_data['kmeans']
        self.scaler = model_data['scaler']
        self.vocabulary = model_data['vocabulary']
        
        print(f"SIFT特征提取器已从 {filepath} 加载")


class ChladniSIFTExtractor(DenseSIFTExtractor):
    """
    专门针对克拉尼图形的SIFT特征提取器
    继承自DenseSIFTExtractor，添加了克拉尼图形特有的预处理
    """
    
    def __init__(self, step_size=8, patch_size=12, vocab_size=150):
        """
        初始化克拉尼图形SIFT提取器
        使用更密集的采样和更大的词汇表，适合克拉尼图形的复杂纹理
        """
        super().__init__(step_size, patch_size, vocab_size)
    
    def preprocess_chladni_image(self, image):
        """
        克拉尼图形专用预处理
        
        Args:
            image: 输入图像
            
        Returns:
            processed_image: 预处理后的图像
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # 确保是uint8类型
        if gray.dtype != np.uint8:
            gray = (gray * 255).astype(np.uint8)
        
        # 直方图均衡化，增强对比度
        gray = cv2.equalizeHist(gray)
        
        # 高斯滤波去噪
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 边缘增强（可选）
        # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        # gray = cv2.filter2D(gray, -1, kernel)
        
        return gray
    
    def extract_dense_sift(self, image):
        """
        重写SIFT提取方法，添加克拉尼图形预处理
        
        Args:
            image: 输入图像
            
        Returns:
            descriptors: SIFT描述符
        """
        # 克拉尼图形预处理
        processed_image = self.preprocess_chladni_image(image)
        
        # 调用父类方法提取SIFT特征
        return super().extract_dense_sift(processed_image)


if __name__ == "__main__":
    # 测试代码
    print("克拉尼图形SIFT特征提取器测试")
    
    # 创建提取器
    extractor = ChladniSIFTExtractor()
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    
    # 提取SIFT特征
    descriptors = extractor.extract_dense_sift(test_image)
    print(f"提取的SIFT描述符数量: {len(descriptors) if len(descriptors) > 0 else 0}")
    
    print("SIFT特征提取器准备就绪！")