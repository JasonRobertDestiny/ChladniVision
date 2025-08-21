# -*- coding: utf-8 -*-
"""
克拉尼图形分类主程序
整合SIFT特征提取和KNN分类的完整流程
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

# 添加utils目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.chladni_preprocessor import ChladniDataPreprocessor
from utils.knn_classifier import ChladniKNNClassifier
from utils.sift_extractor import ChladniSIFTExtractor

class ChladniPatternClassifier:
    """
    克拉尼图形模式分类器
    完整的分类流程：数据加载 -> 特征提取 -> 模型训练 -> 评估预测
    """
    
    def __init__(self, config=None):
        """
        初始化分类器
        
        Args:
            config: 配置字典
        """
        # 默认配置
        self.config = {
            'image_size': (256, 256),
            'sift_params': {
                'step_size': 8,
                'patch_size': 16,
                'vocab_size': 100
            },
            'knn_params': {
                'n_neighbors': 5,
                'weights': 'uniform',
                'metric': 'euclidean'
            },
            'split_params': {
                'test_size': 0.2,
                'val_size': 0.1,
                'random_state': 42
            }
        }
        
        # 更新配置
        if config:
            self.config.update(config)
        
        # 初始化组件
        self.preprocessor = ChladniDataPreprocessor(
            image_size=self.config['image_size'],
            sift_params=self.config['sift_params']
        )
        
        self.classifier = ChladniKNNClassifier(**self.config['knn_params'])
        
        # 数据存储
        self.split_data = None
        self.results = None
        self.is_trained = False
    
    def load_data(self, data_dir):
        """
        加载克拉尼图形数据
        
        Args:
            data_dir: 数据目录路径
        """
        print(f"\n=== 步骤1: 加载数据 ===")
        
        # 检查数据目录
        if not os.path.exists(data_dir):
            raise ValueError(f"数据目录不存在: {data_dir}")
        
        # 加载图像
        num_images = self.preprocessor.load_images_from_directory(data_dir)
        
        if num_images == 0:
            raise ValueError("未找到有效的图像文件")
        
        # 显示数据集信息
        info = self.preprocessor.get_dataset_info()
        print(f"\n数据集信息:")
        print(f"  总图像数: {info['total_images']}")
        print(f"  类别数: {info['num_classes']}")
        print(f"  类别名称: {info['class_names']}")
        print(f"  类别分布: {info['class_distribution']}")
        
        return num_images
    
    def extract_features(self):
        """
        提取SIFT特征
        """
        print(f"\n=== 步骤2: 提取SIFT特征 ===")
        
        # 提取特征
        features = self.preprocessor.extract_sift_features(build_vocabulary=True)
        
        print(f"特征提取完成:")
        print(f"  特征矩阵形状: {features.shape}")
        print(f"  特征维度: {features.shape[1]}")
        
        return features
    
    def split_data(self):
        """
        分割数据集
        """
        print(f"\n=== 步骤3: 分割数据集 ===")
        
        self.split_data = self.preprocessor.split_dataset(**self.config['split_params'])
        
        return self.split_data
    
    def train_model(self, optimize_hyperparams=True):
        """
        训练KNN分类器
        
        Args:
            optimize_hyperparams: 是否优化超参数
        """
        print(f"\n=== 步骤4: 训练KNN分类器 ===")
        
        if self.split_data is None:
            raise ValueError("请先分割数据集")
        
        X_train = self.split_data['X_train']
        y_train = self.split_data['y_train']
        
        # 训练分类器
        self.classifier.train(
            X_train, y_train, 
            class_names=self.preprocessor.class_names,
            optimize=optimize_hyperparams
        )
        
        self.is_trained = True
        
        # 如果有验证集，进行验证
        if self.split_data['X_val'] is not None:
            print("\n验证集评估:")
            val_results = self.classifier.evaluate(
                self.split_data['X_val'], 
                self.split_data['y_val'],
                verbose=True
            )
    
    def evaluate_model(self):
        """
        评估模型性能
        """
        print(f"\n=== 步骤5: 模型评估 ===")
        
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        X_test = self.split_data['X_test']
        y_test = self.split_data['y_test']
        
        # 评估模型
        self.results = self.classifier.evaluate(X_test, y_test, verbose=True)
        
        # 绘制混淆矩阵
        self.classifier.plot_confusion_matrix(
            self.results['confusion_matrix'],
            self.preprocessor.class_names
        )
        
        return self.results
    
    def predict_single_image(self, image_path):
        """
        预测单张图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            prediction: 预测结果字典
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        print(f"\n预测图像: {image_path}")
        
        # 加载和预处理图像
        image = self.preprocessor._load_and_preprocess_image(image_path)
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")
        
        # 提取特征
        features = self.preprocessor.sift_extractor.extract_bow_features([image])
        
        # 预测
        prediction = self.classifier.predict(features)[0]
        probabilities = self.classifier.predict_proba(features)[0]
        
        # 创建结果字典
        result = {
            'predicted_class': prediction,
            'confidence': np.max(probabilities),
            'all_probabilities': dict(zip(self.preprocessor.class_names, probabilities))
        }
        
        print(f"预测结果: {prediction}")
        print(f"置信度: {result['confidence']:.4f}")
        print("各类别概率:")
        for class_name, prob in result['all_probabilities'].items():
            print(f"  {class_name}: {prob:.4f}")
        
        return result
    
    def run_complete_pipeline(self, data_dir, save_model=True, model_save_dir='models'):
        """
        运行完整的分类流程
        
        Args:
            data_dir: 数据目录
            save_model: 是否保存模型
            model_save_dir: 模型保存目录
            
        Returns:
            results: 评估结果
        """
        print("\n" + "="*50)
        print("克拉尼图形分类完整流程")
        print("="*50)
        
        start_time = datetime.now()
        
        try:
            # 1. 加载数据
            self.load_data(data_dir)
            
            # 2. 提取特征
            self.extract_features()
            
            # 3. 分割数据
            self.split_data()
            
            # 4. 训练模型
            self.train_model()
            
            # 5. 评估模型
            results = self.evaluate_model()
            
            # 6. 保存模型
            if save_model:
                self.save_model(model_save_dir)
            
            # 计算总耗时
            total_time = (datetime.now() - start_time).total_seconds()
            
            print(f"\n=== 流程完成 ===")
            print(f"总耗时: {total_time:.2f} 秒")
            print(f"最终测试准确率: {results['accuracy']:.4f}")
            
            return results
            
        except Exception as e:
            print(f"\n流程执行出错: {str(e)}")
            raise
    
    def save_model(self, save_dir):
        """
        保存训练好的模型和预处理器
        
        Args:
            save_dir: 保存目录
        """
        print(f"\n=== 保存模型 ===")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存KNN分类器
        classifier_path = os.path.join(save_dir, 'knn_classifier.pkl')
        self.classifier.save_model(classifier_path)
        
        # 保存预处理数据
        preprocessor_dir = os.path.join(save_dir, 'preprocessor')
        self.preprocessor.save_preprocessed_data(preprocessor_dir)
        
        # 保存配置
        import json
        config_path = os.path.join(save_dir, 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)
        
        print(f"模型已保存到: {save_dir}")
    
    def load_model(self, save_dir):
        """
        加载训练好的模型
        
        Args:
            save_dir: 模型目录
        """
        print(f"\n=== 加载模型 ===")
        
        # 加载配置
        config_path = os.path.join(save_dir, 'config.json')
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        
        # 加载预处理器
        preprocessor_dir = os.path.join(save_dir, 'preprocessor')
        if os.path.exists(preprocessor_dir):
            self.preprocessor.load_preprocessed_data(preprocessor_dir)
        
        # 加载分类器
        classifier_path = os.path.join(save_dir, 'knn_classifier.pkl')
        if os.path.exists(classifier_path):
            self.classifier.load_model(classifier_path)
            self.is_trained = True
        
        print(f"模型已从 {save_dir} 加载")
    
    def create_demo_dataset(self, demo_dir='demo_data', patterns_per_class=10):
        """
        创建演示数据集（生成模拟的克拉尼图形）
        
        Args:
            demo_dir: 演示数据目录
            patterns_per_class: 每个类别的图形数量
        """
        print(f"\n=== 创建演示数据集 ===")
        
        import cv2
        
        os.makedirs(demo_dir, exist_ok=True)
        
        # 定义几种克拉尼图形模式
        patterns = {
            'circular': '圆形模式',
            'radial': '放射模式', 
            'grid': '网格模式',
            'spiral': '螺旋模式'
        }
        
        for pattern_name, pattern_desc in patterns.items():
            pattern_dir = os.path.join(demo_dir, pattern_name)
            os.makedirs(pattern_dir, exist_ok=True)
            
            print(f"生成 {pattern_desc} ({pattern_name})...")
            
            for i in range(patterns_per_class):
                # 生成模拟的克拉尼图形
                image = self._generate_chladni_pattern(pattern_name, i)
                
                # 保存图像
                image_path = os.path.join(pattern_dir, f'{pattern_name}_{i:03d}.png')
                cv2.imwrite(image_path, image)
        
        print(f"演示数据集已创建: {demo_dir}")
        print(f"包含 {len(patterns)} 个类别，每个类别 {patterns_per_class} 张图像")
        
        return demo_dir
    
    def _generate_chladni_pattern(self, pattern_type, seed):
        """
        生成模拟的克拉尼图形
        
        Args:
            pattern_type: 图形类型
            seed: 随机种子
            
        Returns:
            image: 生成的图像
        """
        np.random.seed(seed)
        
        size = self.config['image_size'][0]
        image = np.zeros((size, size), dtype=np.uint8)
        
        # 创建坐标网格
        y, x = np.ogrid[:size, :size]
        center_x, center_y = size // 2, size // 2
        
        if pattern_type == 'circular':
            # 圆形模式
            for r in range(20, size//2, 30):
                mask = np.abs(np.sqrt((x - center_x)**2 + (y - center_y)**2) - r) < 2
                image[mask] = 255
        
        elif pattern_type == 'radial':
            # 放射模式
            angles = np.arctan2(y - center_y, x - center_x)
            for i in range(8):
                angle = i * np.pi / 4
                mask = np.abs(angles - angle) < 0.1
                image[mask] = 255
        
        elif pattern_type == 'grid':
            # 网格模式
            spacing = 25
            image[::spacing, :] = 255
            image[:, ::spacing] = 255
        
        elif pattern_type == 'spiral':
            # 螺旋模式
            for r in range(0, size//2, 2):
                angle = r * 0.1
                x_pos = int(center_x + r * np.cos(angle))
                y_pos = int(center_y + r * np.sin(angle))
                if 0 <= x_pos < size and 0 <= y_pos < size:
                    cv2.circle(image, (x_pos, y_pos), 2, 255, -1)
        
        # 添加噪声
        noise = np.random.normal(0, 10, image.shape)
        image = np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)
        
        # 高斯模糊
        image = cv2.GaussianBlur(image, (3, 3), 0)
        
        return image


def main():
    """
    主函数 - 命令行接口
    """
    parser = argparse.ArgumentParser(description='克拉尼图形分类器')
    parser.add_argument('--data_dir', type=str, help='数据目录路径')
    parser.add_argument('--model_dir', type=str, default='models', help='模型保存/加载目录')
    parser.add_argument('--predict', type=str, help='预测单张图像的路径')
    parser.add_argument('--create_demo', action='store_true', help='创建演示数据集')
    parser.add_argument('--demo_dir', type=str, default='demo_data', help='演示数据目录')
    
    args = parser.parse_args()
    
    # 创建分类器
    classifier = ChladniPatternClassifier()
    
    if args.create_demo:
        # 创建演示数据集
        demo_dir = classifier.create_demo_dataset(args.demo_dir)
        print(f"\n演示数据集已创建: {demo_dir}")
        print("现在可以使用以下命令训练模型:")
        print(f"python chladni_classifier.py --data_dir {demo_dir}")
        return
    
    if args.predict:
        # 预测模式
        if not os.path.exists(args.model_dir):
            print(f"错误: 模型目录不存在 {args.model_dir}")
            print("请先训练模型")
            return
        
        # 加载模型
        classifier.load_model(args.model_dir)
        
        # 预测
        result = classifier.predict_single_image(args.predict)
        
    elif args.data_dir:
        # 训练模式
        if not os.path.exists(args.data_dir):
            print(f"错误: 数据目录不存在 {args.data_dir}")
            return
        
        # 运行完整流程
        results = classifier.run_complete_pipeline(args.data_dir, model_save_dir=args.model_dir)
        
    else:
        # 显示帮助
        parser.print_help()
        print("\n示例用法:")
        print("1. 创建演示数据: python chladni_classifier.py --create_demo")
        print("2. 训练模型: python chladni_classifier.py --data_dir demo_data")
        print("3. 预测图像: python chladni_classifier.py --predict image.jpg")


if __name__ == "__main__":
    main()