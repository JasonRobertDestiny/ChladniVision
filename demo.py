#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的图片分类功能
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 添加项目路径
sys.path.append(os.path.dirname(__file__))

class SimpleImageClassifier:
    """
    简单的图片分类器
    """
    
    def __init__(self):
        self.classifier = KNeighborsClassifier(n_neighbors=3)
        self.data_dir = 'data'
        self.class_names = []
        self.is_trained = False
        self.scaler = StandardScaler()
        self.sift = cv2.SIFT_create()
        self.use_sift_features = True  # 是否使用SIFT特征
    
    def extract_sift_features(self, image, dense_step=10):
        """
        提取Dense SIFT特征
        """
        try:
            # 确保图像是灰度图
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 调整图像大小
            image = cv2.resize(image, (128, 128))
            
            # Dense SIFT: 在规则网格上提取SIFT特征
            keypoints = []
            step = dense_step
            for y in range(step, image.shape[0] - step, step):
                for x in range(step, image.shape[1] - step, step):
                    keypoints.append(cv2.KeyPoint(x, y, step))
            
            # 计算SIFT描述子
            keypoints, descriptors = self.sift.compute(image, keypoints)
            
            if descriptors is not None and len(descriptors) > 0:
                # 将所有描述子展平为一个特征向量
                feature_vector = descriptors.flatten()
                # 如果特征向量太长，可以进行降维或取前N个特征
                if len(feature_vector) > 2000:
                    feature_vector = feature_vector[:2000]
                return feature_vector
            else:
                # 如果SIFT提取失败，回退到像素特征
                return cv2.resize(image, (64, 64)).flatten()
                
        except Exception as e:
            print(f"SIFT特征提取失败: {e}，使用像素特征")
            return cv2.resize(image, (64, 64)).flatten()
    
    def extract_pixel_features(self, image):
        """
        提取简单的像素特征（备用方案）
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (64, 64))
        return image.flatten()
        
    def load_images(self):
        """
        从data文件夹加载图片数据
        """
        images = []
        labels = []
        
        if not os.path.exists(self.data_dir):
            print(f"错误：找不到数据文件夹 {self.data_dir}")
            return None, None
        
        # 获取所有类别文件夹
        self.class_names = [d for d in os.listdir(self.data_dir) 
                           if os.path.isdir(os.path.join(self.data_dir, d))]
        self.class_names.sort()
        
        print(f"找到 {len(self.class_names)} 个类别: {self.class_names}")
        
        # 加载每个类别的图片
        for class_idx, class_name in enumerate(self.class_names):
            class_path = os.path.join(self.data_dir, class_name)
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            print(f"加载类别 {class_name}: {len(image_files)} 张图片")
            
            for image_file in image_files:
                image_path = os.path.join(class_path, image_file)
                image = cv2.imread(image_path)
                if image is not None:
                    # 根据设置选择特征提取方法
                    if self.use_sift_features:
                        feature = self.extract_sift_features(image)
                    else:
                        feature = self.extract_pixel_features(image)
                    
                    images.append(feature)
                    labels.append(class_idx)
        
        return np.array(images), np.array(labels)
    
    def train_model(self):
        """
        训练分类模型
        """
        print("\n🚀 开始训练模型...")
        
        # 加载图片数据
        X, y = self.load_images()
        if X is None or len(X) == 0:
            print("❌ 没有找到图片数据")
            return
        
        print(f"总共加载了 {len(X)} 张图片")
        print(f"特征维度: {X.shape[1]}")
        
        # 特征标准化
        print("正在标准化特征...")
        X_scaled = self.scaler.fit_transform(X)
        
        # 处理小数据集的分割问题
        if len(X) < 10:
            print("⚠️  数据集较小，使用全部数据进行训练和测试")
            X_train = X_test = X_scaled
            y_train = y_test = y
        else:
            try:
                # 尝试分层抽样
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.3, random_state=42, stratify=y
                )
            except ValueError:
                # 如果分层抽样失败，使用普通分割
                print("⚠️  无法进行分层抽样，使用随机分割")
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.3, random_state=42
                )
        
        print(f"训练集: {len(X_train)} 张图片")
        print(f"测试集: {len(X_test)} 张图片")
        
        # 训练模型
        feature_type = "Dense SIFT" if self.use_sift_features else "像素"
        print(f"正在训练KNN分类器（使用{feature_type}特征）...")
        self.classifier.fit(X_train, y_train)
        
        # 测试模型
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n✅ 模型训练完成!")
        print(f"🎯 测试准确率: {accuracy:.2%}")
        
        # 显示详细分类报告
        print("\n📊 详细分类报告:")
        report = classification_report(y_test, y_pred, target_names=self.class_names)
        print(report)
        
        self.is_trained = True
        return accuracy
    
    def predict_image(self, image_path=None):
        """
        预测单张图像
        """
        if not self.is_trained:
            print("❌ 模型尚未训练，请先训练模型")
            return
        
        # 如果没有指定图片路径，让用户输入
        if image_path is None:
            image_path = input("请输入图像路径: ").strip()
        
        if not os.path.exists(image_path):
            print(f"❌ 图像文件不存在: {image_path}")
            return
        
        try:
            # 加载并预处理图像
            image = cv2.imread(image_path)
            if image is None:
                print("❌ 无法读取图像文件")
                return
            
            # 提取特征
            if self.use_sift_features:
                feature = self.extract_sift_features(image)
            else:
                feature = self.extract_pixel_features(image)
            
            # 标准化特征
            feature = self.scaler.transform(feature.reshape(1, -1))
            
            # 预测
            prediction = self.classifier.predict(feature)[0]
            probabilities = self.classifier.predict_proba(feature)[0]
            
            predicted_class = self.class_names[prediction]
            confidence = probabilities[prediction]
            
            print(f"\n🎯 预测结果:")
            print(f"📋 预测类别: {predicted_class}")
            print(f"🎲 置信度: {confidence:.2%}")
            
            print(f"\n📊 各类别概率:")
            for i, (class_name, prob) in enumerate(zip(self.class_names, probabilities)):
                bar = "█" * int(prob * 20)  # 简单的条形图
                print(f"  {class_name:10s}: {prob:.2%} {bar}")
            
            # 显示图像
            self.show_image(image_path, predicted_class, confidence)
            
            return predicted_class, confidence
            
        except Exception as e:
            print(f"❌ 预测过程出错: {str(e)}")
            return None, None
    
    def show_image(self, image_path, predicted_class, confidence):
        """
        显示预测图像和结果
        """
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 创建图像显示
                plt.figure(figsize=(8, 6))
                plt.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
                plt.title(f"预测结果: {predicted_class} (置信度: {confidence:.2%})")
                plt.axis('off')
                plt.tight_layout()
                plt.show()
            
        except Exception as e:
            print(f"显示图像时出错: {str(e)}")
    
    def run_demo(self):
        """
        运行完整的演示流程
        """
        print("\n=== 改进的图片分类演示 ===")
        print("\n🔧 特征提取方法选择:")
        print("1. Dense SIFT特征 (推荐，更准确)")
        print("2. 像素特征 (简单，速度快)")
        
        while True:
            choice = input("\n请选择特征提取方法 (1/2): ").strip()
            if choice == '1':
                self.use_sift_features = True
                print("✅ 已选择Dense SIFT特征")
                break
            elif choice == '2':
                self.use_sift_features = False
                print("✅ 已选择像素特征")
                break
            else:
                print("❌ 无效选择，请输入1或2")
        
        # 1. 训练模型
        print("\n步骤1: 训练模型")
        accuracy = self.train_model()
        
        if accuracy is None:
            print("训练失败，退出演示")
            return
        
        # 2. 预测测试
        print("\n步骤2: 预测测试")
        print("现在可以预测图片了！")
        
        while True:
            choice = input("\n选择操作: 1-预测图片, 0-退出: ").strip()
            if choice == '0':
                break
            elif choice == '1':
                self.predict_image()
            else:
                print("无效选择，请重新输入")
        
        print("\n演示结束！")
    



def main():
    """
    主函数
    """
    # 创建分类器实例
    classifier = SimpleImageClassifier()
    
    # 运行演示
    classifier.run_demo()


if __name__ == "__main__":
    main()