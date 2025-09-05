#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版特征提取模块
提升分类准确率的高级特征提取方法
"""

import cv2
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy import ndimage
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# 修复Windows系统joblib并行处理问题
os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # 限制最大CPU核心数
os.environ['JOBLIB_START_METHOD'] = 'threading'  # 使用threading而不是multiprocessing

class OptimizedFeatureExtractor:
    """优化版特征提取器"""
    
    def __init__(self, image_size=(128, 128)):
        self.image_size = image_size
        self.scaler = StandardScaler()  # 使用StandardScaler替代RobustScaler
        # 优化PCA设置：使用固定维度，确保特征多样性
        self.pca = PCA(n_components=30, random_state=42)  # 30维平衡性能和多样性
        self.is_fitted = False
        
        # 特征配置
        self.feature_config = {
            'enhanced_sift': True,
            'advanced_lbp': True,
            'multi_scale_gradient': True,
            'gabor_texture': True,
            'haralick_features': True,
            'enhanced_statistical': True,
            'frequency_domain': True,
            'shape_features': True,
            'edge_features': True,
            'color_features': True
        }
        
        # Gabor滤波器参数
        self.gabor_kernels = self._create_gabor_kernels()
        
        # SIFT词汇表
        self.sift_vocabulary = None
        self.vocabulary_size = 50
        
    def _create_gabor_kernels(self):
        """创建Gabor滤波器组"""
        kernels = []
        frequencies = [0.1, 0.3, 0.5]
        orientations = [0, 45, 90, 135]
        
        for freq in frequencies:
            for angle in orientations:
                kernel = cv2.getGaborKernel(
                    (21, 21), 3.0, np.deg2rad(angle), 
                    freq, 0.5, 0, ktype=cv2.CV_32F
                )
                kernels.append(kernel)
        
        return kernels
    
    def preprocess_image(self, image):
        """专门为克拉尼图形优化的图像预处理"""
        try:
            # 转换为灰度
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # 确保图像是uint8类型
            if gray.dtype != np.uint8:
                if gray.max() <= 1.0:
                    gray = (gray * 255).astype(np.uint8)
                else:
                    gray = np.clip(gray, 0, 255).astype(np.uint8)
            
            # 调整尺寸 - 使用更好的插值方法
            resized = cv2.resize(gray, self.image_size, interpolation=cv2.INTER_CUBIC)
            
            # 克拉尼图形专用处理：增强对比度
            # 使用更温和的CLAHE设置
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            enhanced = clahe.apply(resized)
            
            # 轻微的高斯模糊去噪，保留边缘细节
            denoised = cv2.GaussianBlur(enhanced, (3, 3), 0.5)
            
            # 二值化处理 - 对克拉尼图形很重要
            _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 形态学操作清理噪点
            kernel = np.ones((2, 2), np.uint8)
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            
            # 归一化到[0,1]范围
            normalized = cleaned.astype(np.float32) / 255.0
            
            return normalized
            
        except Exception as e:
            print(f"克拉尼图形预处理失败: {e}")
            # 备用简单处理
            try:
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image
                
                # 确保正确的数据类型
                if gray.dtype != np.uint8:
                    gray = np.clip(gray * 255 if gray.max() <= 1.0 else gray, 0, 255).astype(np.uint8)
                
                resized = cv2.resize(gray, self.image_size, interpolation=cv2.INTER_AREA)
                normalized = resized.astype(np.float32) / 255.0
                return normalized
            except Exception as e2:
                print(f"备用处理也失败: {e2}")
                return np.zeros(self.image_size, dtype=np.float32)
    
    def extract_enhanced_sift_features(self, image):
        """提取增强的SIFT特征"""
        try:
            img_uint8 = (image * 255).astype(np.uint8)
            
            # 增强SIFT参数
            sift = cv2.SIFT_create(
                nfeatures=100,           # 增加关键点数量
                nOctaveLayers=5,         # 增加 octave 层数
                contrastThreshold=0.02,  # 降低对比度阈值
                edgeThreshold=20,        # 增加边缘阈值
                sigma=1.6                # 高斯模糊参数
            )
            
            keypoints, descriptors = sift.detectAndCompute(img_uint8, None)
            
            if descriptors is not None and len(descriptors) >= 10:
                # 使用MiniBatchKMeans加速聚类
                if self.sift_vocabulary is None:
                    # 首次运行，创建词汇表
                    kmeans = MiniBatchKMeans(
                        n_clusters=min(self.vocabulary_size, len(descriptors)),
                        random_state=42, batch_size=100
                    )
                    kmeans.fit(descriptors)
                    self.sift_vocabulary = kmeans
                else:
                    kmeans = self.sift_vocabulary
                
                # 计算视觉词袋
                hist = np.zeros(self.vocabulary_size)
                for d in descriptors:
                    cluster_idx = kmeans.predict([d])[0]
                    hist[cluster_idx] += 1
                
                # 归一化
                hist = hist / (np.sum(hist) + 1e-7)
                
                # 添加统计特征
                if len(descriptors) > 0:
                    keypoint_info = np.array([
                        len(keypoints),
                        np.mean([kp.size for kp in keypoints]),
                        np.std([kp.size for kp in keypoints]),
                        np.mean([kp.angle for kp in keypoints]),
                        np.std([kp.angle for kp in keypoints]),
                        np.mean([kp.response for kp in keypoints]),
                        np.std([kp.response for kp in keypoints])
                    ])
                else:
                    keypoint_info = np.zeros(7)
                
                return np.concatenate([hist, keypoint_info])
            
            return np.zeros(self.vocabulary_size + 7)
            
        except Exception as e:
            return np.zeros(self.vocabulary_size + 7)
    
    def extract_advanced_lbp_features(self, image):
        """提取高级LBP特征"""
        try:
            img_uint8 = (image * 255).astype(np.uint8)
            
            # 多尺度LBP
            lbp_features = []
            radii = [1, 2, 3, 4]
            n_points = [8, 16, 24, 32]
            
            for radius, n_point in zip(radii, n_points):
                lbp = self._compute_lbp(img_uint8, radius, n_point)
                
                # 计算LBP统计特征
                hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
                hist = hist.astype(np.float32)
                hist = hist / (np.sum(hist) + 1e-7)
                
                # 添加统计量
                stats = [
                    np.mean(lbp),
                    np.std(lbp),
                    np.var(lbp),
                    np.percentile(lbp, 25),
                    np.percentile(lbp, 75)
                ]
                
                lbp_features.extend([hist[::8]])  # 降维
                lbp_features.extend(stats)
            
            return np.concatenate(lbp_features)
            
        except Exception as e:
            return np.zeros(4 * 32 + 4 * 5)  # 4 scales * 32 bins + 4 scales * 5 stats
    
    def _compute_lbp(self, image, radius, n_points):
        """计算LBP特征"""
        h, w = image.shape
        lbp = np.zeros((h, w), dtype=np.uint8)
        
        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                center = image[i, j]
                code = 0
                
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = i + radius * np.cos(angle)
                    y = j + radius * np.sin(angle)
                    
                    if 0 <= x < h and 0 <= y < w:
                        if image[int(x), int(y)] >= center:
                            code += 2**k
                
                lbp[i, j] = code
        
        return lbp
    
    def extract_multi_scale_gradient_features(self, image):
        """提取多尺度梯度特征"""
        try:
            features = []
            
            # 多尺度Sobel算子
            scales = [1, 3, 5]
            for scale in scales:
                # 高斯滤波
                blurred = cv2.GaussianBlur(image, (scale, scale), 0)
                
                # Sobel梯度
                sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
                
                # 梯度幅值和方向
                gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
                gradient_dir = np.arctan2(sobel_y, sobel_x)
                
                # 统计特征
                features.extend([
                    np.mean(gradient_mag),
                    np.std(gradient_mag),
                    np.max(gradient_mag),
                    np.min(gradient_mag),
                    np.percentile(gradient_mag, 25),
                    np.percentile(gradient_mag, 75),
                    np.percentile(gradient_mag, 90),
                    np.percentile(gradient_mag, 95)
                ])
                
                # 方向特征
                features.extend([
                    np.mean(gradient_dir),
                    np.std(gradient_dir),
                    np.max(gradient_dir),
                    np.min(gradient_dir)
                ])
                
                # 梯度直方图
                hist, _ = np.histogram(gradient_dir.ravel(), bins=8, range=(-np.pi, np.pi))
                hist = hist.astype(np.float32)
                hist = hist / (np.sum(hist) + 1e-7)
                features.extend(hist)
            
            return np.array(features)
            
        except Exception as e:
            return np.zeros(3 * 12 + 3 * 8)  # 3 scales * 12 stats + 3 scales * 8 hist bins
    
    def extract_gabor_texture_features(self, image):
        """提取Gabor纹理特征"""
        try:
            img_uint8 = (image * 255).astype(np.uint8)
            features = []
            
            # 应用每个Gabor滤波器
            for kernel in self.gabor_kernels:
                filtered = cv2.filter2D(img_uint8, cv2.CV_8UC3, kernel)
                
                # 计算滤波后的统计特征
                features.extend([
                    np.mean(filtered),
                    np.std(filtered),
                    np.var(filtered),
                    np.max(filtered),
                    np.min(filtered),
                    np.percentile(filtered, 25),
                    np.percentile(filtered, 75)
                ])
                
                # 能量特征
                energy = np.sum(filtered**2)
                features.append(energy)
            
            return np.array(features)
            
        except Exception as e:
            return np.zeros(len(self.gabor_kernels) * 8)
    
    def extract_haralick_features(self, image):
        """提取Haralick纹理特征"""
        try:
            img_uint8 = (image * 255).astype(np.uint8)
            
            # 计算灰度共生矩阵
            distances = [1, 2, 3]
            angles = [0, 45, 90, 135]
            
            haralick_features = []
            
            for distance in distances:
                for angle in angles:
                    # 简化的Haralick特征计算
                    # 使用灰度级减少计算量
                    quantized = (img_uint8 // 32).astype(np.uint8)
                    
                    # 计算共生矩阵
                    glcm = self._compute_glcm(quantized, distance, angle)
                    
                    # 计算Haralick特征
                    features = self._compute_haralick_stats(glcm)
                    haralick_features.extend(features)
            
            return np.array(haralick_features)
            
        except Exception as e:
            return np.zeros(3 * 4 * 6)  # 3 distances * 4 angles * 6 features
    
    def _compute_glcm(self, image, distance, angle):
        """计算灰度共生矩阵"""
        h, w = image.shape
        max_level = 8  # 灰度级
        glcm = np.zeros((max_level, max_level))
        
        # 转换角度为弧度
        angle_rad = np.deg2rad(angle)
        
        # 计算偏移量
        dx = int(round(distance * np.cos(angle_rad)))
        dy = int(round(distance * np.sin(angle_rad)))
        
        # 填充共生矩阵
        for i in range(max(0, -dy), min(h, h - dy)):
            for j in range(max(0, -dx), min(w, w - dx)):
                i2, j2 = i + dy, j + dx
                if 0 <= i2 < h and 0 <= j2 < w:
                    val1 = image[i, j]
                    val2 = image[i2, j2]
                    glcm[val1, val2] += 1
        
        # 归一化
        glcm = glcm / (np.sum(glcm) + 1e-7)
        return glcm
    
    def _compute_haralick_stats(self, glcm):
        """计算Haralick统计特征"""
        try:
            # 计算Haralick特征的简化版本
            h, w = glcm.shape
            
            # 对比度
            contrast = 0
            for i in range(h):
                for j in range(w):
                    contrast += (i - j)**2 * glcm[i, j]
            
            # 能量
            energy = np.sum(glcm**2)
            
            # 熵
            entropy = 0
            for i in range(h):
                for j in range(w):
                    if glcm[i, j] > 0:
                        entropy -= glcm[i, j] * np.log(glcm[i, j] + 1e-7)
            
            # 均值
            mean_i = np.sum(i * glcm[i, j] for i in range(h) for j in range(w))
            mean_j = np.sum(j * glcm[i, j] for i in range(h) for j in range(w))
            
            # 方差
            variance_i = np.sum((i - mean_i)**2 * glcm[i, j] for i in range(h) for j in range(w))
            variance_j = np.sum((j - mean_j)**2 * glcm[i, j] for i in range(h) for j in range(w))
            
            return [contrast, energy, entropy, variance_i, variance_j, (variance_i + variance_j) / 2]
            
        except Exception as e:
            return [0] * 6
    
    def extract_enhanced_statistical_features(self, image):
        """提取增强的统计特征"""
        try:
            features = []
            
            # 基本统计
            features.extend([
                np.mean(image),
                np.std(image),
                np.var(image),
                np.min(image),
                np.max(image),
                np.median(image),
                np.percentile(image, 10),
                np.percentile(image, 25),
                np.percentile(image, 75),
                np.percentile(image, 90)
            ])
            
            # 高阶统计
            features.extend([
                self._skewness(image),
                self._kurtosis(image),
                self._entropy(image)
            ])
            
            # 形状特征
            features.extend([
                np.sum(image > np.mean(image)) / image.size,  # 超过均值的像素比例
                np.sum(image < np.mean(image)) / image.size,  # 低于均值的像素比例
                np.sum(image > np.percentile(image, 75)) / image.size,  # 高亮度像素比例
                np.sum(image < np.percentile(image, 25)) / image.size   # 低亮度像素比例
            ])
            
            # 局部统计特征
            h, w = image.shape
            blocks = [(0, h//2, 0, w//2), (0, h//2, w//2, w), 
                     (h//2, h, 0, w//2), (h//2, h, w//2, w)]
            
            for i1, i2, j1, j2 in blocks:
                block = image[i1:i2, j1:j2]
                features.extend([
                    np.mean(block),
                    np.std(block),
                    np.max(block),
                    np.min(block)
                ])
            
            return np.array(features)
            
        except Exception as e:
            return np.zeros(10 + 3 + 4 + 4 * 4)  # 基本统计 + 高阶统计 + 形状特征 + 局部统计
    
    def _skewness(self, data):
        """计算偏度"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std)**3)
    
    def _kurtosis(self, data):
        """计算峰度"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std)**4) - 3
    
    def _entropy(self, data):
        """计算熵"""
        hist, _ = np.histogram(data, bins=256, range=(0, 1))
        hist = hist / (np.sum(hist) + 1e-7)
        entropy = 0
        for p in hist:
            if p > 0:
                entropy -= p * np.log(p + 1e-7)
        return entropy
    
    def extract_frequency_domain_features(self, image):
        """提取频域特征"""
        try:
            # 傅里叶变换
            f_transform = np.fft.fft2(image)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.abs(f_shift)
            
            features = []
            
            # 频域能量分布
            h, w = magnitude_spectrum.shape
            center_h, center_w = h // 2, w // 2
            
            # 环形能量分布
            radii = [5, 10, 15, 20, 25, 30, 35, 40]
            for radius in radii:
                if radius < min(center_h, center_w):
                    y, x = np.ogrid[:h, :w]
                    mask = (x - center_w)**2 + (y - center_h)**2 <= radius**2
                    
                    if radius > 5:
                        inner_mask = (x - center_w)**2 + (y - center_h)**2 <= (radius-5)**2
                        mask = mask & ~inner_mask
                    
                    energy = np.sum(magnitude_spectrum[mask])
                    features.append(energy)
            
            # 象限能量分布
            total_energy = np.sum(magnitude_spectrum)
            if total_energy > 0:
                q1 = np.sum(magnitude_spectrum[:center_h, :center_w]) / total_energy
                q2 = np.sum(magnitude_spectrum[:center_h, center_w:]) / total_energy
                q3 = np.sum(magnitude_spectrum[center_h:, :center_w]) / total_energy
                q4 = np.sum(magnitude_spectrum[center_h:, center_w:]) / total_energy
                features.extend([q1, q2, q3, q4])
            else:
                features.extend([0, 0, 0, 0])
            
            # 频谱统计特征
            features.extend([
                np.mean(magnitude_spectrum),
                np.std(magnitude_spectrum),
                np.max(magnitude_spectrum),
                np.percentile(magnitude_spectrum, 90),
                np.percentile(magnitude_spectrum, 95)
            ])
            
            return np.array(features)
            
        except Exception as e:
            return np.zeros(8 + 4 + 5)  # 环形能量 + 象限能量 + 频谱统计
    
    def extract_shape_features(self, image):
        """提取形状特征"""
        try:
            img_uint8 = (image * 255).astype(np.uint8)
            
            # 二值化
            _, binary = cv2.threshold(img_uint8, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 形态学操作
            kernel = np.ones((3, 3), np.uint8)
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # 查找轮廓
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            features = []
            
            if contours:
                # 最大轮廓
                largest_contour = max(contours, key=cv2.contourArea)
                
                # 轮廓特征
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter**2)
                else:
                    circularity = 0
                
                # 边界框
                x, y, w, h = cv2.boundingRect(largest_contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # 凸包
                hull = cv2.convexHull(largest_contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                
                features.extend([
                    area, perimeter, circularity, aspect_ratio, solidity,
                    len(contours),  # 轮廓数量
                    w, h,           # 边界框尺寸
                    len(largest_contour)  # 轮廓点数
                ])
            else:
                features = [0] * 9
            
            return np.array(features)
            
        except Exception as e:
            return np.zeros(9)
    
    def extract_edge_features(self, image):
        """提取边缘特征"""
        try:
            img_uint8 = (image * 255).astype(np.uint8)
            
            features = []
            
            # 多阈值边缘检测
            thresholds = [50, 100, 150, 200]
            for threshold in thresholds:
                edges = cv2.Canny(img_uint8, threshold//2, threshold)
                
                # 边缘统计
                edge_density = np.sum(edges > 0) / edges.size
                edge_length = np.sum(edges > 0)
                
                features.extend([edge_density, edge_length])
            
            # Sobel边缘强度
            sobel_x = cv2.Sobel(img_uint8, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(img_uint8, cv2.CV_64F, 0, 1, ksize=3)
            sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
            
            features.extend([
                np.mean(sobel_mag),
                np.std(sobel_mag),
                np.max(sobel_mag),
                np.percentile(sobel_mag, 75),
                np.percentile(sobel_mag, 90)
            ])
            
            return np.array(features)
            
        except Exception as e:
            return np.zeros(4 * 2 + 5)  # 4 thresholds * 2 features + 5 sobel features
    
    def extract_color_features(self, image):
        """提取颜色特征（如果是彩色图像）"""
        try:
            # 如果是灰度图像，返回零向量
            if len(image.shape) == 2:
                return np.zeros(15)
            
            # 转换为不同颜色空间
            img_uint8 = (image * 255).astype(np.uint8)
            
            # RGB空间
            r, g, b = cv2.split(img_uint8)
            
            # HSV空间
            hsv = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            features = []
            
            # RGB统计
            for channel in [r, g, b]:
                features.extend([
                    np.mean(channel),
                    np.std(channel),
                    np.max(channel),
                    np.min(channel)
                ])
            
            # HSV统计
            features.extend([
                np.mean(h), np.std(h),
                np.mean(s), np.std(s),
                np.mean(v), np.std(v)
            ])
            
            # 颜色矩
            features.extend([
                np.mean(img_uint8),
                np.std(img_uint8),
                np.var(img_uint8)
            ])
            
            return np.array(features)
            
        except Exception as e:
            return np.zeros(15)
    
    def extract_all_features(self, images):
        """提取所有特征"""
        print("🔍 提取优化版多模态特征...")
        
        all_features = []
        
        for i, image in enumerate(images):
            try:
                # 预处理
                processed_image = self.preprocess_image(image)
                
                feature_vector = []
                
                # 提取各种特征
                if self.feature_config['enhanced_sift']:
                    features = self.extract_enhanced_sift_features(processed_image)
                    feature_vector.extend(features)
                
                if self.feature_config['advanced_lbp']:
                    features = self.extract_advanced_lbp_features(processed_image)
                    feature_vector.extend(features)
                
                if self.feature_config['multi_scale_gradient']:
                    features = self.extract_multi_scale_gradient_features(processed_image)
                    feature_vector.extend(features)
                
                if self.feature_config['gabor_texture']:
                    features = self.extract_gabor_texture_features(processed_image)
                    feature_vector.extend(features)
                
                if self.feature_config['haralick_features']:
                    features = self.extract_haralick_features(processed_image)
                    feature_vector.extend(features)
                
                if self.feature_config['enhanced_statistical']:
                    features = self.extract_enhanced_statistical_features(processed_image)
                    feature_vector.extend(features)
                
                if self.feature_config['frequency_domain']:
                    features = self.extract_frequency_domain_features(processed_image)
                    feature_vector.extend(features)
                
                if self.feature_config['shape_features']:
                    features = self.extract_shape_features(processed_image)
                    feature_vector.extend(features)
                
                if self.feature_config['edge_features']:
                    features = self.extract_edge_features(processed_image)
                    feature_vector.extend(features)
                
                if self.feature_config['color_features']:
                    features = self.extract_color_features(processed_image)
                    feature_vector.extend(features)
                
                # 清理异常值
                feature_vector = np.nan_to_num(feature_vector, nan=0.0)
                all_features.append(feature_vector)
                
            except Exception as e:
                print(f"特征提取失败 (图像 {i}): {e}")
                # 使用零向量
                total_dims = self._get_total_feature_dimensions()
                all_features.append(np.zeros(total_dims))
        
        features = np.array(all_features)
        features = np.nan_to_num(features, nan=0.0)
        
        print(f"✅ 特征提取完成，维度: {features.shape[1]}")
        
        # 保存特征维度信息
        self.n_features = features.shape[1]
        
        return features
    
    def _get_total_feature_dimensions(self):
        """计算总特征维度"""
        dimensions = 0
        
        if self.feature_config['enhanced_sift']:
            dimensions += 50 + 7  # SIFT词汇表 + 关键点信息
        
        if self.feature_config['advanced_lbp']:
            dimensions += 4 * 32 + 4 * 5  # 4 scales * 32 bins + 4 scales * 5 stats
        
        if self.feature_config['multi_scale_gradient']:
            dimensions += 3 * 12 + 3 * 8  # 3 scales * 12 stats + 3 scales * 8 hist bins
        
        if self.feature_config['gabor_texture']:
            dimensions += len(self.gabor_kernels) * 8  # 12 kernels * 8 features
        
        if self.feature_config['haralick_features']:
            dimensions += 3 * 4 * 6  # 3 distances * 4 angles * 6 features
        
        if self.feature_config['enhanced_statistical']:
            dimensions += 10 + 3 + 4 + 4 * 4  # 基本统计 + 高阶统计 + 形状特征 + 局部统计
        
        if self.feature_config['frequency_domain']:
            dimensions += 8 + 4 + 5  # 环形能量 + 象限能量 + 频谱统计
        
        if self.feature_config['shape_features']:
            dimensions += 9  # 形状特征
        
        if self.feature_config['edge_features']:
            dimensions += 4 * 2 + 5  # 4 thresholds * 2 features + 5 sobel features
        
        if self.feature_config['color_features']:
            dimensions += 15  # 颜色特征
        
        return dimensions
    
    def optimize_features(self, X_features):
        """特征优化处理"""
        print("🔧 优化特征处理...")
        
        # 特征标准化
        X_scaled = self.scaler.fit_transform(X_features)
        
        # PCA降维
        X_pca = self.pca.fit_transform(X_scaled)
        
        print(f"✅ 特征优化完成: {X_scaled.shape[1]} → {X_pca.shape[1]}")
        
        self.is_fitted = True
        
        return X_pca
    
    def transform_features(self, X_features):
        """转换新特征（用于预测）"""
        if not self.is_fitted:
            raise ValueError("特征提取器未训练，请先调用 optimize_features")
        
        # 标准化
        X_scaled = self.scaler.transform(X_features)
        
        # PCA降维
        X_pca = self.pca.transform(X_scaled)
        
        return X_pca