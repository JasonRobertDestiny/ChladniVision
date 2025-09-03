#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChladniVision Pro 基本测试
验证系统核心功能是否正常
"""

import sys
import os
import unittest
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from chladni_vision_pro import ChladniVisionPro
    import numpy as np
    import cv2
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保安装了所有依赖: pip install -r requirements.txt")
    sys.exit(1)

class TestChladniVisionPro(unittest.TestCase):
    """ChladniVision Pro 基本功能测试"""
    
    def setUp(self):
        """测试前设置"""
        self.system = ChladniVisionPro()
        self.test_image_path = "data/data/600Hz/600hz_001.png"
        
    def test_import(self):
        """测试模块导入"""
        self.assertIsNotNone(self.system)
        
    def test_data_loading(self):
        """测试数据加载"""
        if os.path.exists("data/data_augmented/"):
            X_images, y, paths = self.system.load_dataset("data/data_augmented/")
            self.assertIsNotNone(X_images)
            self.assertGreater(len(X_images), 0)
            
    def test_feature_extraction(self):
        """测试特征提取"""
        # 创建测试图像
        test_image = np.random.rand(64, 64).astype(np.float32)
        test_images = [test_image]
        
        features = self.system.extract_features(test_images)
        self.assertIsNotNone(features)
        self.assertEqual(features.shape[0], 1)
        
    def test_model_loading(self):
        """测试模型加载"""
        if os.path.exists("models/demo_model.pkl"):
            success = self.system.load_model("models/demo_model.pkl")
            self.assertTrue(success)
            
    def test_prediction(self):
        """测试预测功能"""
        if os.path.exists(self.test_image_path) and os.path.exists("models/demo_model.pkl"):
            # 加载模型
            self.system.load_model("models/demo_model.pkl")
            
            # 预测
            result = self.system.predict_and_visualize(self.test_image_path, save_result=False)
            self.assertIsNotNone(result)
            self.assertIn('predicted_class', result)
            self.assertIn('confidence', result)

def run_tests():
    """运行测试"""
    print("=" * 60)
    print("ChladniVision Pro 功能测试")
    print("=" * 60)
    
    # 检查必要文件
    required_files = [
        "src/chladni_vision_pro.py",
        "src/run_chladni_pro_windows.py",
        "requirements.txt"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ 缺少必要文件:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    print("✅ 所有必要文件存在")
    
    # 检查数据目录
    data_dirs = ["data/data/", "data/data_augmented/"]
    existing_dirs = [d for d in data_dirs if os.path.exists(d)]
    
    if existing_dirs:
        print(f"✅ 数据目录存在: {', '.join(existing_dirs)}")
    else:
        print("⚠️  未找到数据目录")
    
    # 检查模型文件
    model_files = ["models/demo_model.pkl", "models/pro_enhanced_model.pkl"]
    existing_models = [m for m in model_files if os.path.exists(m)]
    
    if existing_models:
        print(f"✅ 模型文件存在: {', '.join(existing_models)}")
    else:
        print("⚠️  未找到模型文件")
    
    # 运行单元测试
    print("\n🧪 运行单元测试...")
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestChladniVisionPro)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("\n🎉 所有测试通过!")
        return True
    else:
        print(f"\n❌ {len(result.failures)} 个测试失败, {len(result.errors)} 个错误")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)