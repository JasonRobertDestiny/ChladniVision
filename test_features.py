#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征提取方法对比测试
"""

import cv2
import numpy as np
from demo import SimpleImageClassifier
import os

def compare_features():
    """
    对比不同特征提取方法的效果
    """
    print("\n=== 特征提取方法对比测试 ===")
    
    # 测试图片路径
    test_image = "data/1100Hz/1100hz_001.png"
    
    if not os.path.exists(test_image):
        print(f"❌ 测试图片不存在: {test_image}")
        return
    
    # 加载图片
    image = cv2.imread(test_image)
    if image is None:
        print("❌ 无法读取测试图片")
        return
    
    print(f"📸 测试图片: {test_image}")
    print(f"图片尺寸: {image.shape}")
    
    # 创建分类器实例
    classifier = SimpleImageClassifier()
    
    # 测试SIFT特征
    print("\n🔍 测试Dense SIFT特征:")
    sift_features = classifier.extract_sift_features(image)
    print(f"SIFT特征维度: {sift_features.shape}")
    print(f"SIFT特征范围: [{sift_features.min():.3f}, {sift_features.max():.3f}]")
    print(f"SIFT特征均值: {sift_features.mean():.3f}")
    
    # 测试像素特征
    print("\n🖼️  测试像素特征:")
    pixel_features = classifier.extract_pixel_features(image)
    print(f"像素特征维度: {pixel_features.shape}")
    print(f"像素特征范围: [{pixel_features.min():.3f}, {pixel_features.max():.3f}]")
    print(f"像素特征均值: {pixel_features.mean():.3f}")
    
    # 特征对比
    print("\n📊 特征对比:")
    print(f"SIFT特征维度: {sift_features.shape[0]}")
    print(f"像素特征维度: {pixel_features.shape[0]}")
    print(f"维度比例: {sift_features.shape[0] / pixel_features.shape[0]:.2f}")
    
    print("\n✅ 特征提取测试完成！")

if __name__ == "__main__":
    compare_features()