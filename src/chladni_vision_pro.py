#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChladniVision Pro - 优化版系统
增强特征提取、改进可视化、优化算法选择
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib
from tqdm import tqdm
from datetime import datetime
import argparse
from collections import Counter
import warnings
import pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')

# 设置更美观的绘图样式
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Windows系统编码修复
def safe_print(text):
    """完全兼容Windows的安全打印函数"""
    try:
        # 移除所有emoji和特殊字符
        clean_text = text
        emoji_replacements = {
            '🎵': '[音乐]', '🚀': '[启动]', '📁': '[文件夹]', '❌': '[失败]', '✅': '[成功]',
            '🤖': '[AI]', '🔍': '[搜索]', '📊': '[图表]', '🎯': '[目标]', '🎲': '[概率]',
            '📈': '[上升]', '💾': '[保存]', '⏱️': '[时间]', '🎉': '[完成]', '👋': '[再见]',
            '📷': '[相机]', '📖': '[说明]', '🔧': '[设置]', '⭐': '[星级]', '🔬': '[科学]',
            '🎨': '[艺术]', '🌟': '[亮点]', '💡': '[提示]', '🎪': '[演示]', '🏆': '[奖杯]',
            '📋': '[列表]', '🔄': '[交互]', '✨': '[闪亮]', '🔔': '[通知]', '📱': '[手机]',
            '💻': '[电脑]', '🌐': '[网络]', '🎮': '[游戏]', '🎁': '[礼物]', '🎊': '[派对]'
        }
        
        for emoji, replacement in emoji_replacements.items():
            clean_text = clean_text.replace(emoji, replacement)
        
        print(clean_text)
    except Exception:
        # 如果仍有问题，使用最基础的ASCII字符
        try:
            ascii_text = text.encode('ascii', 'ignore').decode('ascii')
            print(ascii_text)
        except:
            print("输出信息无法显示")

# 设置matplotlib字体
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

class ChladniVisionPro:
    """
    ChladniVision Pro - 优化版系统
    """
    
    def __init__(self):
        """初始化系统"""
        self.model = None
        self.scaler = RobustScaler()  # 使用更鲁棒的缩放器
        self.class_names = []
        self.is_trained = False
        self.model_info = {}
        self.feature_extractor = None
        self.image_size = (128, 128)  # 提高图像分辨率
        self.output_dir = "output"
        self.feature_history = []
        self.prediction_history = []
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 特征提取配置
        self.feature_config = {
            'sift': True,
            'lbp': True,
            'gradient': True,
            'texture': True,
            'statistical': True,
            'frequency': True
        }
    
    def welcome_message(self):
        """显示欢迎信息"""
        safe_print("=" * 70)
        safe_print("[音乐] ChladniVision Pro - 优化版系统")
        safe_print("   增强特征提取 | 智能算法选择 | 精美可视化")
        safe_print("=" * 70)
        safe_print("")
        safe_print("🌟 核心优化:")
        safe_print("   ✅ 多模态特征提取 (SIFT+LBP+梯度+纹理+频域)")
        safe_print("   ✅ 智能算法选择与超参数优化")
        safe_print("   ✅ 鲁棒特征缩放与降维")
        safe_print("   ✅ 实时预测与批量处理")
        safe_print("   ✅ 交互式可视化界面")
        safe_print("   ✅ 详细性能分析与报告")
        safe_print("")
    
    def load_dataset(self, data_dir):
        """加载数据集"""
        safe_print(f"[文件夹] 加载数据集: {data_dir}")
        
        if not os.path.exists(data_dir):
            safe_print("[失败] 数据目录不存在")
            return None, None, None
        
        images = []
        labels = []
        paths = []
        
        # 检查数据集结构
        if 'train' in os.listdir(data_dir) and 'test' in os.listdir(data_dir):
            # train/test 结构
            for split in ['train', 'test']:
                split_dir = os.path.join(data_dir, split)
                if not os.path.exists(split_dir):
                    continue
                
                safe_print(f"   处理 {split} 数据...")
                freq_dirs = [d for d in os.listdir(split_dir) 
                           if os.path.isdir(os.path.join(split_dir, d))]
                
                for freq in freq_dirs:
                    freq_dir = os.path.join(split_dir, freq)
                    self._load_images_from_dir(freq_dir, freq, images, labels, paths)
        else:
            # 频率目录结构
            safe_print("   处理频率目录...")
            freq_dirs = [d for d in os.listdir(data_dir) 
                       if os.path.isdir(os.path.join(data_dir, d))]
            
            for freq in freq_dirs:
                freq_dir = os.path.join(data_dir, freq)
                self._load_images_from_dir(freq_dir, freq, images, labels, paths)
        
        if not images:
            safe_print("[失败] 未找到图像文件")
            return None, None, None
        
        safe_print(f"   [成功] 加载 {len(images)} 张图像")
        class_counts = Counter(labels)
        safe_print("   类别分布:")
        for class_name, count in class_counts.items():
            safe_print(f"      {class_name}: {count} 张")
        
        self.class_names = sorted(list(set(labels)))
        return np.array(images), np.array(labels), paths
    
    def _load_images_from_dir(self, dir_path, label, images, labels, paths):
        """从目录加载图像"""
        if not os.path.exists(dir_path):
            return
        
        image_files = [f for f in os.listdir(dir_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for filename in tqdm(image_files, desc=f"      {label}", leave=False):
            image_path = os.path.join(dir_path, filename)
            
            try:
                # 读取和预处理图像
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                # 转换为灰度
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # 调整尺寸
                resized = cv2.resize(gray, self.image_size, interpolation=cv2.INTER_AREA)
                
                # 自适应直方图均衡化
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(resized)
                
                # 归一化
                normalized = enhanced.astype(np.float32) / 255.0
                
                images.append(normalized)
                labels.append(label)
                paths.append(image_path)
                
            except Exception as e:
                continue
    
    def extract_sift_features(self, image):
        """提取SIFT特征"""
        try:
            # 转换为uint8
            img_uint8 = (image * 255).astype(np.uint8)
            
            # 创建SIFT检测器
            sift = cv2.SIFT_create(nfeatures=50, nOctaveLayers=3, 
                                   contrastThreshold=0.03, edgeThreshold=10)
            
            # 检测关键点和计算描述符
            keypoints, descriptors = sift.detectAndCompute(img_uint8, None)
            
            if descriptors is not None and len(descriptors) > 0:
                # 使用K-means聚类生成视觉词袋
                if len(descriptors) >= 10:
                    kmeans = KMeans(n_clusters=min(10, len(descriptors)), random_state=42)
                    kmeans.fit(descriptors)
                    
                    # 计算直方图
                    hist = np.zeros(10)
                    for d in descriptors:
                        cluster_idx = kmeans.predict([d])[0]
                        hist[cluster_idx] += 1
                    
                    # 归一化直方图
                    hist = hist / (np.sum(hist) + 1e-7)
                    return hist
            
            return np.zeros(10)
            
        except Exception:
            return np.zeros(10)
    
    def extract_lbp_features(self, image):
        """提取LBP (局部二值模式) 特征"""
        try:
            img_uint8 = (image * 255).astype(np.uint8)
            
            # LBP参数
            radius = 3
            n_points = 8 * radius
            
            # 计算LBP
            lbp = np.zeros_like(img_uint8)
            for i in range(radius, img_uint8.shape[0] - radius):
                for j in range(radius, img_uint8.shape[1] - radius):
                    center = img_uint8[i, j]
                    code = 0
                    for k in range(n_points):
                        angle = 2 * np.pi * k / n_points
                        x = i + radius * np.cos(angle)
                        y = j + radius * np.sin(angle)
                        
                        if 0 <= x < img_uint8.shape[0] and 0 <= y < img_uint8.shape[1]:
                            if img_uint8[int(x), int(y)] >= center:
                                code += 2**k
                    
                    lbp[i, j] = code
            
            # 计算LBP直方图
            hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
            hist = hist.astype(np.float32)
            hist = hist / (np.sum(hist) + 1e-7)
            
            # 降维到32维
            return hist[::8]  # 每8个bin取一个
            
        except Exception:
            return np.zeros(32)
    
    def extract_gradient_features(self, image):
        """提取梯度特征"""
        try:
            # Sobel梯度
            sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            
            # 梯度幅值和方向
            gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
            gradient_dir = np.arctan2(sobel_y, sobel_x)
            
            # 统计特征
            features = [
                np.mean(gradient_mag),
                np.std(gradient_mag),
                np.max(gradient_mag),
                np.min(gradient_mag),
                np.mean(gradient_dir),
                np.std(gradient_dir),
                np.percentile(gradient_mag, 25),
                np.percentile(gradient_mag, 75),
                np.percentile(gradient_mag, 90),
                np.percentile(gradient_mag, 95)
            ]
            
            # 梯度直方图 (8个方向)
            hist, _ = np.histogram(gradient_dir.ravel(), bins=8, range=(-np.pi, np.pi))
            hist = hist.astype(np.float32)
            hist = hist / (np.sum(hist) + 1e-7)
            
            return np.concatenate([features, hist])
            
        except Exception:
            return np.zeros(18)
    
    def extract_texture_features(self, image):
        """提取纹理特征"""
        try:
            img_uint8 = (image * 255).astype(np.uint8)
            
            # 计算灰度共生矩阵特征
            # 简化的纹理特征
            features = []
            
            # 不同方向的纹理特征
            angles = [0, 45, 90, 135]
            for angle in angles:
                # 计算特定方向的梯度
                rad = np.deg2rad(angle)
                kernel_x = np.cos(rad)
                kernel_y = np.sin(rad)
                
                gradient = cv2.filter2D(img_uint8, -1, np.array([[kernel_x, kernel_y]]))
                features.extend([
                    np.mean(gradient),
                    np.std(gradient),
                    np.max(gradient),
                    np.min(gradient)
                ])
            
            return np.array(features)
            
        except Exception:
            return np.zeros(16)
    
    def extract_statistical_features(self, image):
        """提取统计特征"""
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
                np.percentile(image, 25),
                np.percentile(image, 75)
            ])
            
            # 高阶统计
            features.extend([
                np.mean((image - np.mean(image))**3),  # 偏度
                np.mean((image - np.mean(image))**4),  # 峰度
                np.sum(image > np.mean(image)) / image.size,  # 超过均值的像素比例
                np.sum(image < np.mean(image)) / image.size   # 低于均值的像素比例
            ])
            
            # 分位数特征
            for p in [10, 20, 30, 40, 60, 70, 80, 90]:
                features.append(np.percentile(image, p))
            
            return np.array(features)
            
        except Exception:
            return np.zeros(22)
    
    def extract_frequency_features(self, image):
        """提取频域特征"""
        try:
            # 傅里叶变换
            f_transform = np.fft.fft2(image)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.abs(f_shift)
            
            # 频域统计特征
            features = []
            
            # 不同频率环的能量
            h, w = magnitude_spectrum.shape
            center_h, center_w = h // 2, w // 2
            
            radii = [5, 10, 15, 20, 25, 30]
            for radius in radii:
                if radius < min(center_h, center_w):
                    # 创建环形掩码
                    y, x = np.ogrid[:h, :w]
                    mask = (x - center_w)**2 + (y - center_h)**2 <= radius**2
                    
                    if radius > 5:
                        inner_mask = (x - center_w)**2 + (y - center_h)**2 <= (radius-5)**2
                        mask = mask & ~inner_mask
                    
                    energy = np.sum(magnitude_spectrum[mask])
                    features.append(energy)
            
            # 总能量和能量分布
            total_energy = np.sum(magnitude_spectrum)
            features.append(total_energy)
            
            if total_energy > 0:
                features.extend([
                    np.sum(magnitude_spectrum[:center_h, :center_w]) / total_energy,  # 左上象限
                    np.sum(magnitude_spectrum[:center_h, center_w:]) / total_energy,  # 右上象限
                    np.sum(magnitude_spectrum[center_h:, :center_w]) / total_energy,  # 左下象限
                    np.sum(magnitude_spectrum[center_h:, center_w:]) / total_energy   # 右下象限
                ])
            else:
                features.extend([0, 0, 0, 0])
            
            return np.array(features)
            
        except Exception:
            return np.zeros(11)
    
    def extract_features(self, images):
        """提取综合特征"""
        safe_print("[搜索] 提取多模态图像特征...")
        
        features = []
        
        for i, image in enumerate(tqdm(images, desc="   提取特征")):
            try:
                feature_vector = []
                
                # 提取各种特征
                if self.feature_config['sift']:
                    sift_features = self.extract_sift_features(image)
                    feature_vector.extend(sift_features)
                
                if self.feature_config['lbp']:
                    lbp_features = self.extract_lbp_features(image)
                    feature_vector.extend(lbp_features)
                
                if self.feature_config['gradient']:
                    gradient_features = self.extract_gradient_features(image)
                    feature_vector.extend(gradient_features)
                
                if self.feature_config['texture']:
                    texture_features = self.extract_texture_features(image)
                    feature_vector.extend(texture_features)
                
                if self.feature_config['statistical']:
                    statistical_features = self.extract_statistical_features(image)
                    feature_vector.extend(statistical_features)
                
                if self.feature_config['frequency']:
                    frequency_features = self.extract_frequency_features(image)
                    feature_vector.extend(frequency_features)
                
                # 清理异常值
                feature_vector = np.nan_to_num(feature_vector, nan=0.0)
                features.append(feature_vector)
                
            except Exception as e:
                # 使用零向量
                total_features = sum([
                    10 if self.feature_config['sift'] else 0,
                    32 if self.feature_config['lbp'] else 0,
                    18 if self.feature_config['gradient'] else 0,
                    16 if self.feature_config['texture'] else 0,
                    22 if self.feature_config['statistical'] else 0,
                    11 if self.feature_config['frequency'] else 0
                ])
                dummy_feature = np.zeros(total_features)
                features.append(dummy_feature)
        
        features = np.array(features)
        features = np.nan_to_num(features, nan=0.0)
        
        safe_print(f"   [成功] 特征维度: {features.shape[1]}")
        
        # 记录特征统计信息
        self.feature_history.append({
            'timestamp': datetime.now(),
            'feature_shape': features.shape,
            'feature_config': self.feature_config.copy()
        })
        
        return features
    
    def optimize_features(self, X_features):
        """优化特征处理"""
        safe_print("[🔧] 优化特征处理...")
        
        # 特征标准化
        X_scaled = self.scaler.fit_transform(X_features)
        
        # PCA降维 (保留95%的方差)
        pca = PCA(n_components=0.95, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        safe_print(f"   [成功] PCA降维: {X_scaled.shape[1]} → {X_pca.shape[1]}")
        
        return X_pca, pca
    
    def optimize_hyperparameters(self, X_train, y_train, model_name):
        """优化超参数"""
        safe_print(f"[🔧] 优化 {model_name} 超参数...")
        
        if model_name == 'KNN':
            param_grid = {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            }
            base_model = KNeighborsClassifier()
            
        elif model_name == 'RandomForest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            base_model = RandomForestClassifier(random_state=42)
            
        elif model_name == 'SVM':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'poly', 'sigmoid']
            }
            base_model = SVC(random_state=42)
            
        elif model_name == 'MLP':
            param_grid = {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            }
            base_model = MLPClassifier(random_state=42, max_iter=1000)
            
        else:
            return None
        
        # 网格搜索
        grid_search = GridSearchCV(
            base_model, param_grid, cv=5, scoring='accuracy', 
            n_jobs=-1, verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        safe_print(f"   [成功] 最佳参数: {grid_search.best_params_}")
        safe_print(f"   [成功] 最佳得分: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def select_best_model(self, X_train, y_train):
        """选择最佳模型"""
        safe_print("[🤖] 智能算法选择与优化...")
        
        models = {
            'KNN': KNeighborsClassifier(),
            'RandomForest': RandomForestClassifier(random_state=42),
            'SVM': SVC(random_state=42, probability=True),
            'MLP': MLPClassifier(random_state=42, max_iter=1000),
            'GradientBoosting': GradientBoostingClassifier(random_state=42)
        }
        
        best_model = None
        best_score = 0
        best_name = ""
        model_scores = {}
        
        for name, base_model in models.items():
            try:
                # 优化超参数
                optimized_model = self.optimize_hyperparameters(X_train, y_train, name)
                
                if optimized_model is None:
                    # 如果优化失败，使用默认模型
                    optimized_model = base_model
                
                # 交叉验证
                cv_scores = cross_val_score(optimized_model, X_train, y_train, cv=5, scoring='accuracy')
                mean_score = cv_scores.mean()
                std_score = cv_scores.std()
                
                model_scores[name] = {
                    'mean_score': mean_score,
                    'std_score': std_score,
                    'model': optimized_model
                }
                
                safe_print(f"   {name}: {mean_score:.4f} (±{std_score:.4f})")
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_model = optimized_model
                    best_name = name
                    
            except Exception as e:
                safe_print(f"   {name}: 训练失败 - {e}")
                model_scores[name] = {'mean_score': 0, 'std_score': 0, 'model': None}
        
        safe_print(f"   [🌟] 选择模型: {best_name} (准确率: {best_score:.4f})")
        
        self.model_info = {
            'name': best_name,
            'validation_score': best_score,
            'model': best_model,
            'all_scores': model_scores
        }
        
        return best_model
    
    def train(self, data_dir, model_path='chladni_pro_model.pkl'):
        """训练模型"""
        safe_print(f"\n[🚀] 开始训练模型...")
        safe_print(f"   数据目录: {data_dir}")
        
        start_time = datetime.now()
        
        try:
            # 加载数据
            X_images, y, paths = self.load_dataset(data_dir)
            if X_images is None:
                return False
            
            # 检查数据量是否足够
            class_counts = Counter(y)
            min_samples = min(class_counts.values())
            
            if min_samples < 2:
                safe_print(f"[❌] 训练失败: 某些类别样本太少 (最少: {min_samples} 个)")
                return False
            
            # 提取特征
            X_features = self.extract_features(X_images)
            
            # 特征优化
            X_optimized, pca = self.optimize_features(X_features)
            
            # 选择最佳模型
            self.model = self.select_best_model(X_optimized, y)
            
            # 训练最终模型
            safe_print("[📊] 训练最终模型...")
            self.model.fit(X_optimized, y)
            
            # 评估模型
            safe_print("[📈] 评估模型性能...")
            y_pred = self.model.predict(X_optimized)
            accuracy = accuracy_score(y, y_pred)
            
            # 保存PCA用于预测
            self.pca = pca
            
            # 生成增强的可视化
            self.generate_enhanced_visualizations(y, y_pred, paths, X_optimized)
            
            # 设置训练标志并保存模型
            self.is_trained = True
            self.save_model(model_path)
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            safe_print(f"\n[🎉] 训练完成!")
            safe_print(f"   [⏱️] 训练时间: {training_time:.2f} 秒")
            safe_print(f"   [📊] 训练准确率: {accuracy:.4f}")
            safe_print(f"   [💾] 模型已保存: {model_path}")
            
            return True
            
        except Exception as e:
            safe_print(f"[❌] 训练失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_enhanced_visualizations(self, y_true, y_pred, paths, X_features):
        """生成增强的可视化结果"""
        safe_print("[🎨] 生成增强可视化结果...")
        
        try:
            # 1. 增强混淆矩阵
            self.create_enhanced_confusion_matrix(y_true, y_pred)
            
            # 2. 模型性能对比
            self.create_model_comparison_chart()
            
            # 3. 特征分析
            self.create_feature_analysis(X_features)
            
            # 4. 增强样本展示
            self.create_enhanced_sample_grid(paths, y_true, y_pred)
            
            # 5. 训练历史
            self.create_training_summary(y_true, y_pred)
            
            safe_print(f"   [✅] 增强可视化结果已保存到 {self.output_dir}/ 目录")
            
        except Exception as e:
            safe_print(f"   [❌] 可视化生成失败: {e}")
    
    def create_enhanced_confusion_matrix(self, y_true, y_pred):
        """创建增强的混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        
        # 计算百分比
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 绝对数量
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   annot_kws={"size": 12}, ax=ax1)
        ax1.set_title('混淆矩阵 (绝对数量)', fontsize=14, pad=20)
        ax1.set_xlabel('预测类别', fontsize=12)
        ax1.set_ylabel('真实类别', fontsize=12)
        
        # 百分比
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Reds',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   annot_kws={"size": 12}, ax=ax2)
        ax2.set_title('混淆矩阵 (百分比)', fontsize=14, pad=20)
        ax2.set_xlabel('预测类别', fontsize=12)
        ax2.set_ylabel('真实类别', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/enhanced_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_model_comparison_chart(self):
        """创建模型对比图表"""
        if 'all_scores' not in self.model_info:
            return
        
        model_scores = self.model_info['all_scores']
        names = []
        scores = []
        errors = []
        
        for name, score_info in model_scores.items():
            if score_info['model'] is not None:
                names.append(name)
                scores.append(score_info['mean_score'])
                errors.append(score_info['std_score'])
        
        if not names:
            return
        
        plt.figure(figsize=(12, 8))
        
        # 创建柱状图
        bars = plt.bar(names, scores, yerr=errors, capsize=5, alpha=0.7)
        
        # 着色
        colors = plt.cm.Set3(np.linspace(0, 1, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # 添加数值标签
        for i, (score, error) in enumerate(zip(scores, errors)):
            plt.text(i, score + error + 0.01, f'{score:.3f}', 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.title('模型性能对比', fontsize=16, pad=20)
        plt.xlabel('算法', fontsize=12)
        plt.ylabel('交叉验证准确率', fontsize=12)
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3)
        
        # 突出显示最佳模型
        best_idx = np.argmax(scores)
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_feature_analysis(self, X_features):
        """创建特征分析图表"""
        plt.figure(figsize=(15, 10))
        
        # 特征分布
        plt.subplot(2, 2, 1)
        feature_means = np.mean(X_features, axis=0)
        plt.hist(feature_means, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('特征均值分布', fontsize=14)
        plt.xlabel('特征值')
        plt.ylabel('频次')
        
        # 特征方差
        plt.subplot(2, 2, 2)
        feature_vars = np.var(X_features, axis=0)
        plt.hist(feature_vars, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.title('特征方差分布', fontsize=14)
        plt.xlabel('特征值')
        plt.ylabel('频次')
        
        # 特征相关性热图
        plt.subplot(2, 2, 3)
        if X_features.shape[1] > 20:
            # 如果特征太多，只显示前20个
            corr_matrix = np.corrcoef(X_features[:, :20].T)
        else:
            corr_matrix = np.corrcoef(X_features.T)
        
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0, 
                   square=True, cbar_kws={"shrink": 0.8})
        plt.title('特征相关性热图', fontsize=14)
        
        # PCA可视化
        plt.subplot(2, 2, 4)
        if X_features.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_features)
            
            # 按类别着色
            colors = plt.cm.Set3(np.linspace(0, 1, len(self.class_names)))
            for i, class_name in enumerate(self.class_names):
                mask = np.array([self.class_names.index(label) == i for label in self.class_names * (len(X_pca) // len(self.class_names))])
                if len(mask) == len(X_pca):
                    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                              c=[colors[i]], label=class_name, alpha=0.7)
            
            plt.title('PCA 2D可视化', fontsize=14)
            plt.xlabel('第一主成分')
            plt.ylabel('第二主成分')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/feature_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_enhanced_sample_grid(self, paths, y_true, y_pred):
        """创建增强的样本展示网格"""
        try:
            samples_per_class = 4
            sample_images = []
            sample_labels = []
            sample_confidences = []
            
            for class_name in self.class_names:
                class_indices = [i for i, label in enumerate(y_true) if label == class_name]
                selected_indices = class_indices[:samples_per_class]
                
                for idx in selected_indices:
                    if len(sample_images) < 16:  # 最多显示16张
                        sample_images.append(paths[idx])
                        correct = y_true[idx] == y_pred[idx]
                        sample_labels.append(f"{y_true[idx]}→{y_pred[idx]}")
                        sample_confidences.append(correct)
            
            if sample_images:
                fig, axes = plt.subplots(4, 4, figsize=(16, 16))
                axes = axes.flatten()
                
                for i, (img_path, label, correct) in enumerate(zip(sample_images, sample_labels, sample_confidences)):
                    if i < len(axes):
                        img = cv2.imread(img_path)
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            axes[i].imshow(img)
                            
                            # 根据正确性设置标题颜色
                            title_color = 'green' if correct else 'red'
                            axes[i].set_title(label, fontsize=12, color=title_color, fontweight='bold')
                            axes[i].axis('off')
                            
                            # 添加边框
                            border_color = 'green' if correct else 'red'
                            for spine in axes[i].spines.values():
                                spine.set_edgecolor(border_color)
                                spine.set_linewidth(3)
                
                # 隐藏多余的子图
                for i in range(len(sample_images), len(axes)):
                    axes[i].axis('off')
                
                plt.tight_layout()
                plt.savefig(f'{self.output_dir}/enhanced_sample_predictions.png', dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            safe_print(f"   [❌] 增强样本网格生成失败: {e}")
    
    def create_training_summary(self, y_true, y_pred):
        """创建训练摘要"""
        try:
            # 生成详细的分类报告
            report = classification_report(y_true, y_pred, target_names=self.class_names, 
                                         digits=4, zero_division=0)
            
            # 创建摘要图表
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.axis('off')
            
            # 文本内容
            summary_text = f"""
ChladniVision Pro 训练摘要
{'='*50}

模型信息:
• 最佳算法: {self.model_info.get('name', 'Unknown')}
• 验证准确率: {self.model_info.get('validation_score', 0):.4f}
• 类别数量: {len(self.class_names)}
• 样本总数: {len(y_true)}

性能指标:
• 训练准确率: {accuracy_score(y_true, y_pred):.4f}
• 模型类型: {type(self.model).__name__}

特征配置:
• SIFT特征: {'✅' if self.feature_config['sift'] else '❌'}
• LBP特征: {'✅' if self.feature_config['lbp'] else '❌'}
• 梯度特征: {'✅' if self.feature_config['gradient'] else '❌'}
• 纹理特征: {'✅' if self.feature_config['texture'] else '❌'}
• 统计特征: {'✅' if self.feature_config['statistical'] else '❌'}
• 频域特征: {'✅' if self.feature_config['frequency'] else '❌'}

训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
                   fontsize=12, verticalalignment='top', fontfamily='monospace')
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/training_summary.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 保存文本报告
            with open(f'{self.output_dir}/detailed_classification_report.txt', 'w', encoding='utf-8') as f:
                f.write("ChladniVision Pro 详细分类报告\n")
                f.write("="*60 + "\n\n")
                f.write(summary_text)
                f.write("\n\n详细分类报告:\n")
                f.write(report)
                
        except Exception as e:
            safe_print(f"   [❌] 训练摘要生成失败: {e}")
    
    def predict_and_visualize(self, image_path, save_result=True):
        """预测并可视化结果"""
        if not self.is_trained:
            safe_print("[❌] 模型尚未训练")
            return None
        
        try:
            safe_print(f"[🎯] 预测图像: {image_path}")
            
            # 预处理图像
            image = cv2.imread(image_path)
            if image is None:
                safe_print(f"[❌] 无法读取图像: {image_path}")
                return None
            
            original_image = image.copy()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, self.image_size, interpolation=cv2.INTER_AREA)
            
            # 自适应直方图均衡化
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(resized)
            normalized = enhanced.astype(np.float32) / 255.0
            
            # 提取特征
            feature_vector = []
            if self.feature_config['sift']:
                feature_vector.extend(self.extract_sift_features(normalized))
            if self.feature_config['lbp']:
                feature_vector.extend(self.extract_lbp_features(normalized))
            if self.feature_config['gradient']:
                feature_vector.extend(self.extract_gradient_features(normalized))
            if self.feature_config['texture']:
                feature_vector.extend(self.extract_texture_features(normalized))
            if self.feature_config['statistical']:
                feature_vector.extend(self.extract_statistical_features(normalized))
            if self.feature_config['frequency']:
                feature_vector.extend(self.extract_frequency_features(normalized))
            
            feature_vector = np.nan_to_num(feature_vector, nan=0.0).reshape(1, -1)
            
            # 特征优化
            feature_scaled = self.scaler.transform(feature_vector)
            feature_pca = self.pca.transform(feature_scaled)
            
            # 预测
            prediction = self.model.predict(feature_pca)[0]
            probabilities = self.model.predict_proba(feature_pca)[0]
            confidence = np.max(probabilities)
            
            result = {
                'predicted_class': prediction,
                'confidence': confidence,
                'probabilities': dict(zip(self.class_names, probabilities))
            }
            
            # 记录预测历史
            self.prediction_history.append({
                'timestamp': datetime.now(),
                'image_path': image_path,
                'prediction': prediction,
                'confidence': confidence,
                'probabilities': result['probabilities'].copy()
            })
            
            # 显示结果
            safe_print(f"   [🎯] 预测结果: {prediction}")
            safe_print(f"   [🎲] 置信度: {confidence:.4f}")
            
            safe_print("   [📊] 各类别概率:")
            sorted_probs = sorted(result['probabilities'].items(), 
                                key=lambda x: x[1], reverse=True)
            
            for class_name, prob in sorted_probs:
                bar_length = int(prob * 40)
                bar = "█" * bar_length + "░" * (40 - bar_length)
                safe_print(f"      {class_name:8s}: {prob:.4f} {bar}")
            
            # 保存可视化结果
            if save_result:
                self.save_enhanced_prediction_visualization(original_image, result, image_path)
            
            return result
            
        except Exception as e:
            safe_print(f"[❌] 预测失败: {e}")
            return None
    
    def save_enhanced_prediction_visualization(self, image, result, image_path):
        """保存增强的预测可视化结果"""
        try:
            fig = plt.figure(figsize=(16, 10))
            
            # 创建更复杂的布局
            gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], width_ratios=[1, 1, 1])
            
            # 原始图像
            ax1 = fig.add_subplot(gs[0, 0])
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ax1.imshow(image_rgb)
            ax1.set_title('原始图像', fontsize=14, fontweight='bold')
            ax1.axis('off')
            
            # 预测结果概率分布
            ax2 = fig.add_subplot(gs[0, 1])
            probs = list(result['probabilities'].values())
            classes = list(result['probabilities'].keys())
            bars = ax2.bar(classes, probs, color='skyblue', alpha=0.7)
            
            # 突出显示预测结果
            pred_idx = classes.index(result['predicted_class'])
            bars[pred_idx].set_color('red')
            bars[pred_idx].set_alpha(0.9)
            
            ax2.set_title('预测概率分布', fontsize=14, fontweight='bold')
            ax2.set_ylabel('概率')
            ax2.set_ylim(0, 1)
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
            
            # 添加概率值标签
            for i, (class_name, prob) in enumerate(result['probabilities'].items()):
                ax2.text(i, prob + 0.01, f'{prob:.3f}', 
                        ha='center', va='bottom', fontweight='bold')
            
            # 预测信息面板
            ax3 = fig.add_subplot(gs[0, 2])
            ax3.axis('off')
            
            info_text = f"""
预测结果: {result['predicted_class']}
置信度: {result['confidence']:.4f}

模型信息:
算法: {self.model_info.get('name', 'Unknown')}
验证分数: {self.model_info.get('validation_score', 0):.4f}

图像信息:
文件名: {os.path.basename(image_path)}
尺寸: {image.shape[:2]}
预测时间: {datetime.now().strftime('%H:%M:%S')}
            """
            
            ax3.text(0.1, 0.9, info_text, transform=ax3.transAxes, 
                    fontsize=12, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            # 置信度历史
            ax4 = fig.add_subplot(gs[1, :])
            if len(self.prediction_history) > 1:
                confidences = [p['confidence'] for p in self.prediction_history[-10:]]
                times = [p['timestamp'].strftime('%H:%M:%S') for p in self.prediction_history[-10:]]
                
                ax4.plot(times, confidences, 'o-', color='green', linewidth=2, markersize=6)
                ax4.set_title('最近10次预测的置信度变化', fontsize=12, fontweight='bold')
                ax4.set_ylabel('置信度')
                ax4.set_ylim(0, 1)
                plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, '更多预测后将显示置信度历史', 
                        ha='center', va='center', transform=ax4.transAxes, fontsize=12)
                ax4.axis('off')
            
            plt.tight_layout()
            
            # 生成输出文件名
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = f"{self.output_dir}/enhanced_prediction_{image_name}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            safe_print(f"   [💾] 增强预测结果已保存: {output_path}")
            
        except Exception as e:
            safe_print(f"   [❌] 增强可视化保存失败: {e}")
    
    def save_model(self, model_path):
        """保存模型"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'pca': self.pca,
            'class_names': self.class_names,
            'image_size': self.image_size,
            'model_info': self.model_info,
            'feature_config': self.feature_config,
            'is_trained': self.is_trained,
            'feature_history': self.feature_history
        }
        
        joblib.dump(model_data, model_path)
    
    def load_model(self, model_path):
        """加载模型"""
        if not os.path.exists(model_path):
            safe_print(f"[❌] 模型文件不存在: {model_path}")
            return False
        
        try:
            model_data = joblib.load(model_path)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.pca = model_data['pca']
            self.class_names = model_data['class_names']
            self.image_size = model_data['image_size']
            self.model_info = model_data.get('model_info', {})
            self.feature_config = model_data.get('feature_config', self.feature_config)
            self.is_trained = model_data['is_trained']
            self.feature_history = model_data.get('feature_history', [])
            
            safe_print("[✅] 模型加载成功")
            safe_print(f"   模型类型: {self.model_info.get('name', 'Unknown')}")
            safe_print(f"   支持类别: {len(self.class_names)}")
            safe_print(f"   特征配置: {sum(self.feature_config.values())} 种特征")
            return True
            
        except Exception as e:
            safe_print(f"[❌] 模型加载失败: {e}")
            return False
    
    def interactive_mode(self):
        """交互式模式"""
        safe_print("\n[🎯] 交互式模式")
        safe_print("   输入图像路径进行预测，输入 'quit' 退出")
        safe_print(f"   输出目录: {self.output_dir}/")
        safe_print("   可用命令:")
        safe_print("     - 'stats': 显示预测统计")
        safe_print("     - 'history': 显示预测历史")
        safe_print("     - 'batch': 批量预测模式")
        safe_print("     - 'quit': 退出")
        
        while True:
            user_input = input("\n[📷] 请输入命令或图像路径: ").strip()
            
            if user_input.lower() == 'quit':
                safe_print("[👋] 退出交互模式")
                break
            
            elif user_input.lower() == 'stats':
                self.show_prediction_stats()
            
            elif user_input.lower() == 'history':
                self.show_prediction_history()
            
            elif user_input.lower() == 'batch':
                self.batch_prediction_mode()
            
            elif os.path.exists(user_input):
                self.predict_and_visualize(user_input, save_result=True)
            else:
                safe_print("[❌] 文件不存在或命令无效，请重新输入")
    
    def show_prediction_stats(self):
        """显示预测统计"""
        if not self.prediction_history:
            safe_print("   暂无预测记录")
            return
        
        safe_print("\n[📊] 预测统计:")
        safe_print(f"   总预测次数: {len(self.prediction_history)}")
        
        # 类别分布
        class_counts = Counter([p['prediction'] for p in self.prediction_history])
        safe_print("   预测类别分布:")
        for class_name, count in class_counts.items():
            safe_print(f"      {class_name}: {count} 次")
        
        # 置信度统计
        confidences = [p['confidence'] for p in self.prediction_history]
        safe_print(f"   平均置信度: {np.mean(confidences):.4f}")
        safe_print(f"   最高置信度: {np.max(confidences):.4f}")
        safe_print(f"   最低置信度: {np.min(confidences):.4f}")
    
    def show_prediction_history(self):
        """显示预测历史"""
        if not self.prediction_history:
            safe_print("   暂无预测记录")
            return
        
        safe_print("\n[📖] 最近10次预测历史:")
        for i, record in enumerate(self.prediction_history[-10:], 1):
            safe_print(f"   {i}. {record['timestamp'].strftime('%H:%M:%S')} - "
                       f"{record['prediction']} ({record['confidence']:.4f})")
    
    def batch_prediction_mode(self):
        """批量预测模式"""
        safe_print("\n[📁] 批量预测模式")
        safe_print("   输入包含图像的目录路径")
        
        dir_path = input("请输入目录路径: ").strip()
        
        if not os.path.exists(dir_path):
            safe_print("[❌] 目录不存在")
            return
        
        # 查找图像文件
        image_files = []
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, file))
        
        if not image_files:
            safe_print("[❌] 目录中未找到图像文件")
            return
        
        safe_print(f"   找到 {len(image_files)} 张图像")
        
        # 批量预测
        results = []
        for image_path in tqdm(image_files, desc="   批量预测"):
            result = self.predict_and_visualize(image_path, save_result=True)
            if result:
                results.append({
                    'path': image_path,
                    'prediction': result['predicted_class'],
                    'confidence': result['confidence']
                })
        
        # 保存批量预测结果
        if results:
            batch_summary_path = f"{self.output_dir}/batch_prediction_summary.txt"
            with open(batch_summary_path, 'w', encoding='utf-8') as f:
                f.write("批量预测结果摘要\n")
                f.write("="*50 + "\n\n")
                f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"总图像数: {len(image_files)}\n")
                f.write(f"成功预测: {len(results)}\n\n")
                
                f.write("预测结果:\n")
                for result in results:
                    f.write(f"{result['path']}: {result['prediction']} "
                           f"({result['confidence']:.4f})\n")
                
                # 统计
                pred_counts = Counter([r['prediction'] for r in results])
                f.write("\n类别统计:\n")
                for class_name, count in pred_counts.items():
                    f.write(f"{class_name}: {count}\n")
            
            safe_print(f"   [💾] 批量预测结果已保存: {batch_summary_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='ChladniVision Pro - 优化版系统')
    parser.add_argument('--train', action='store_true', help='训练模式')
    parser.add_argument('--data_dir', type=str, help='数据目录路径')
    parser.add_argument('--predict', type=str, help='预测图像路径')
    parser.add_argument('--model', type=str, default='chladni_pro_model.pkl', 
                       help='模型文件路径')
    parser.add_argument('--interactive', action='store_true', help='交互式模式')
    parser.add_argument('--output_dir', type=str, default='output', 
                       help='输出目录')
    parser.add_argument('--demo', action='store_true', help='演示模式')
    
    args = parser.parse_args()
    
    # 创建系统实例
    system = ChladniVisionPro()
    system.output_dir = args.output_dir
    system.welcome_message()
    
    if args.demo:
        # 演示模式
        safe_print("\n[演示] 演示模式")
        safe_print("   将使用增强数据集进行快速演示...")
        
        # 使用增强数据集训练
        success = system.train('data_augmented/', 'demo_model.pkl')
        if success:
            safe_print("\n[目标] 演示预测:")
            # 演示预测
            demo_image = 'data/600Hz/600hz_001.png'
            if os.path.exists(demo_image):
                system.predict_and_visualize(demo_image, save_result=True)
            
            # 进入交互模式
            system.interactive_mode()
    
    elif args.train:
        # 训练模式
        if not args.data_dir:
            safe_print("[❌] 训练模式需要指定 --data_dir")
            return
        
        success = system.train(args.data_dir, args.model)
        if success and args.interactive:
            system.interactive_mode()
    
    elif args.predict:
        # 预测模式
        if not system.load_model(args.model):
            return
        
        system.predict_and_visualize(args.predict, save_result=True)
    
    elif args.interactive:
        # 交互式模式
        if not system.load_model(args.model):
            safe_print("   请先训练模型或指定正确的模型路径")
            return
        
        system.interactive_mode()
    
    else:
        # 显示使用说明
        safe_print("\n[📖] 使用说明:")
        safe_print("   1. 演示模式 (快速体验):")
        safe_print("      python chladni_vision_pro.py --demo")
        safe_print()
        safe_print("   2. 训练模型:")
        safe_print("      python chladni_vision_pro.py --train --data_dir data/")
        safe_print()
        safe_print("   3. 预测图像:")
        safe_print("      python chladni_vision_pro.py --predict image.png")
        safe_print()
        safe_print("   4. 交互式模式:")
        safe_print("      python chladni_vision_pro.py --interactive")
        safe_print()
        safe_print("   5. 训练后进入交互模式:")
        safe_print("      python chladni_vision_pro.py --train --data_dir data/ --interactive")
        safe_print()
        safe_print("[🔧] 输出文件:")
        safe_print("   - output/enhanced_confusion_matrix.png (增强混淆矩阵)")
        safe_print("   - output/model_comparison.png (模型性能对比)")
        safe_print("   - output/feature_analysis.png (特征分析)")
        safe_print("   - output/enhanced_sample_predictions.png (增强样本展示)")
        safe_print("   - output/enhanced_prediction_*.png (增强预测结果)")
        safe_print("   - output/detailed_classification_report.txt (详细报告)")

if __name__ == "__main__":
    main()