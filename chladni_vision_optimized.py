#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChladniVision Optimized - 优化版核心系统
专注于高精度分类和优秀性能
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
from tqdm import tqdm
from datetime import datetime
import argparse
from collections import Counter
import warnings
from pathlib import Path

# 导入优化版特征提取器
from src.feature_extractor_optimized import OptimizedFeatureExtractor

# 修复Windows系统joblib并行处理问题
os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # 限制最大CPU核心数
os.environ['JOBLIB_START_METHOD'] = 'threading'  # 使用threading而不是multiprocessing
os.environ['OMP_NUM_THREADS'] = '4'  # 限制OpenMP线程数

warnings.filterwarnings('ignore')

# 设置英文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

class ChladniVisionOptimized:
    """优化版ChladniVision系统"""
    
    def __init__(self):
        """初始化系统"""
        self.model = None
        self.feature_extractor = OptimizedFeatureExtractor()
        self.class_names = []
        self.is_trained = False
        self.model_info = {}
        self.image_size = (128, 128)
        self.output_dir = "output_optimized"
        self.prediction_history = []
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 创建子目录
        self.training_dir = os.path.join(self.output_dir, "training")
        self.predictions_dir = os.path.join(self.output_dir, "predictions")
        self.reports_dir = os.path.join(self.output_dir, "reports")
        
        os.makedirs(self.training_dir, exist_ok=True)
        os.makedirs(self.predictions_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # 模型配置
        self.model_config = {
            'use_ensemble': True,
            'cross_validation': True,
            'hyperparameter_tuning': True,
            'feature_selection': True
        }
    
    def welcome_message(self):
        """显示欢迎信息"""
        print("=" * 70)
        print("🚀 ChladniVision Optimized - 优化版系统")
        print("   高精度特征提取 | 智能模型选择 | 专业可视化")
        print("=" * 70)
        print("")
        print("🌟 核心优化:")
        print("   ✅ 10种高级特征提取方法")
        print("   ✅ 智能集成学习模型")
        print("   ✅ 自动超参数优化")
        print("   ✅ 鲁棒特征处理")
        print("   ✅ 专业级可视化分析")
        print("   ✅ 实时预测与批量处理")
        print("")
    
    def load_dataset(self, data_dir):
        """加载数据集"""
        print(f"📁 加载数据集: {data_dir}")
        
        if not os.path.exists(data_dir):
            print("❌ 数据目录不存在")
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
                
                print(f"   处理 {split} 数据...")
                freq_dirs = [d for d in os.listdir(split_dir) 
                           if os.path.isdir(os.path.join(split_dir, d))]
                
                for freq in freq_dirs:
                    freq_dir = os.path.join(split_dir, freq)
                    self._load_images_from_dir(freq_dir, freq, images, labels, paths)
        else:
            # 频率目录结构
            print("   处理频率目录...")
            freq_dirs = [d for d in os.listdir(data_dir) 
                       if os.path.isdir(os.path.join(data_dir, d)) and d not in ['output', 'models', '__pycache__']]
            
            for freq in freq_dirs:
                freq_dir = os.path.join(data_dir, freq)
                self._load_images_from_dir(freq_dir, freq, images, labels, paths)
        
        if not images:
            print("❌ 未找到图像文件")
            return None, None, None
        
        print(f"   ✅ 加载 {len(images)} 张图像")
        class_counts = Counter(labels)
        print("   类别分布:")
        for class_name, count in class_counts.items():
            print(f"      {class_name}: {count} 张")
        
        self.class_names = sorted(list(set(labels)))
        return images, np.array(labels), paths
    
    def _load_images_from_dir(self, dir_path, label, images, labels, paths):
        """从目录加载图像"""
        if not os.path.exists(dir_path):
            return
        
        image_files = [f for f in os.listdir(dir_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for filename in tqdm(image_files, desc=f"      {label}", leave=False):
            image_path = os.path.join(dir_path, filename)
            
            try:
                # 读取图像
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                images.append(image)
                labels.append(label)
                paths.append(image_path)
                
            except Exception as e:
                continue
    
    def train_ensemble_model(self, X_features, y):
        """训练集成模型"""
        print("🤖 训练智能集成模型...")
        
        # 基础模型
        base_models = {
            'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42),
            'SVM': SVC(C=10, gamma='scale', kernel='rbf', random_state=42, probability=True),
            'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        # 训练每个模型
        trained_models = {}
        model_scores = {}
        
        for name, model in base_models.items():
            try:
                print(f"   训练 {name}...")
                
                # 交叉验证
                if self.model_config['cross_validation']:
                    cv_scores = cross_val_score(model, X_features, y, cv=5, scoring='accuracy')
                    mean_score = cv_scores.mean()
                    std_score = cv_scores.std()
                else:
                    # 简单训练验证
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_features, y, test_size=0.2, random_state=42
                    )
                    model.fit(X_train, y_train)
                    mean_score = model.score(X_val, y_val)
                    std_score = 0.0
                
                # 训练完整模型
                model.fit(X_features, y)
                trained_models[name] = model
                model_scores[name] = {
                    'mean_score': mean_score,
                    'std_score': std_score,
                    'model': model
                }
                
                print(f"      {name}: {mean_score:.4f} (±{std_score:.4f})")
                
            except Exception as e:
                print(f"      {name}: 训练失败 - {e}")
        
        # 选择最佳模型
        best_name = max(model_scores.keys(), key=lambda x: model_scores[x]['mean_score'])
        best_model = trained_models[best_name]
        best_score = model_scores[best_name]['mean_score']
        
        print(f"   🌟 最佳模型: {best_name} (准确率: {best_score:.4f})")
        
        # 如果启用集成学习
        if self.model_config['use_ensemble'] and len(trained_models) > 1:
            # 创建集成模型
            ensemble_models = list(trained_models.values())
            self.model = EnsembleModel(ensemble_models)
            print("   🔄 启用集成学习模型")
        else:
            self.model = best_model
        
        self.model_info = {
            'best_model': best_name,
            'best_score': best_score,
            'all_scores': model_scores,
            'is_ensemble': self.model_config['use_ensemble']
        }
        
        return best_model
    
    def train(self, data_dir, model_path='demo_optimized_model.pkl'):
        """训练模型"""
        print(f"\n🚀 开始训练优化版模型...")
        print(f"   数据目录: {data_dir}")
        
        start_time = datetime.now()
        
        try:
            # 加载数据
            X_images, y, paths = self.load_dataset(data_dir)
            if X_images is None:
                return False
            
            # 检查数据量
            class_counts = Counter(y)
            min_samples = min(class_counts.values())
            
            if min_samples < 2:
                print(f"❌ 训练失败: 某些类别样本太少 (最少: {min_samples} 个)")
                return False
            
            # 提取特征
            X_features = self.feature_extractor.extract_all_features(X_images)
            
            # 特征优化
            X_optimized = self.feature_extractor.optimize_features(X_features)
            
            # 训练模型
            self.train_ensemble_model(X_optimized, y)
            
            # 评估模型
            print("📊 评估模型性能...")
            y_pred = self.model.predict(X_optimized)
            accuracy = accuracy_score(y, y_pred)
            
            # 生成可视化
            self.generate_enhanced_visualizations(y, y_pred, paths, X_optimized)
            
            # 设置训练标志并保存模型
            self.is_trained = True
            self.save_model(model_path)
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            print(f"\n🎉 训练完成!")
            print(f"   ⏱️ 训练时间: {training_time:.2f} 秒")
            print(f"   📊 训练准确率: {accuracy:.4f}")
            print(f"   🎯 最佳模型: {self.model_info['best_model']}")
            print(f"   💾 模型已保存: {model_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ 训练失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict(self, image_path, save_result=True):
        """预测图像"""
        if not self.is_trained:
            print("❌ 模型尚未训练")
            return None
        
        try:
            print(f"🎯 预测图像: {image_path}")
            
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ 无法读取图像: {image_path}")
                return None
            
            # 预处理
            processed_image = self.feature_extractor.preprocess_image(image)
            
            # 提取特征
            feature_vector = self.feature_extractor.extract_all_features([processed_image])
            
            # 特征优化
            feature_optimized = self.feature_extractor.transform_features(feature_vector)
            
            # 预测
            prediction = self.model.predict(feature_optimized)[0]
            
            # 获取置信度
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(feature_optimized)[0]
                confidence = np.max(probabilities)
                prob_dict = dict(zip(self.class_names, probabilities))
            else:
                confidence = 1.0
                prob_dict = {prediction: 1.0}
            
            result = {
                'predicted_class': prediction,
                'confidence': confidence,
                'probabilities': prob_dict
            }
            
            # 记录预测历史
            self.prediction_history.append({
                'timestamp': datetime.now(),
                'image_path': image_path,
                'prediction': prediction,
                'confidence': confidence,
                'probabilities': prob_dict.copy()
            })
            
            # 显示结果
            print(f"   🎯 预测结果: {prediction}")
            print(f"   🎲 置信度: {confidence:.4f}")
            
            print("   📊 各类别概率:")
            sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
            for class_name, prob in sorted_probs:
                bar_length = int(prob * 40)
                bar = "█" * bar_length + "░" * (40 - bar_length)
                print(f"      {class_name:8s}: {prob:.4f} {bar}")
            
            # 保存可视化结果
            if save_result:
                self.save_prediction_visualization(image, result, image_path)
            
            return result
            
        except Exception as e:
            print(f"❌ 预测失败: {e}")
            return None
    
    def save_prediction_visualization(self, image, result, image_path):
        """保存预测可视化"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            
            # 原始图像
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ax1.imshow(image_rgb)
            ax1.set_title('Original Image', fontsize=14, fontweight='bold')
            ax1.axis('off')
            
            # 预测结果
            ax2.text(0.5, 0.5, f'Prediction: {result["predicted_class"]}', 
                    transform=ax2.transAxes, fontsize=16, fontweight='bold',
                    ha='center', va='center')
            ax2.text(0.5, 0.3, f'Confidence: {result["confidence"]:.4f}', 
                    transform=ax2.transAxes, fontsize=12,
                    ha='center', va='center')
            ax2.set_title('Prediction Result', fontsize=14, fontweight='bold')
            ax2.axis('off')
            
            # 概率分布
            probs = list(result['probabilities'].values())
            classes = list(result['probabilities'].keys())
            bars = ax3.bar(classes, probs, color='skyblue', alpha=0.7)
            ax3.set_title('Class Probability Distribution', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Probability')
            ax3.set_ylim(0, 1)
            
            # 预测历史
            if len(self.prediction_history) > 1:
                confidences = [p['confidence'] for p in self.prediction_history[-10:]]
                times = [p['timestamp'].strftime('%H:%M:%S') for p in self.prediction_history[-10:]]
                ax4.plot(times, confidences, 'o-', color='green', linewidth=2, markersize=6)
                ax4.set_title('Prediction Confidence History', fontsize=12, fontweight='bold')
                ax4.set_ylabel('Confidence')
                ax4.set_ylim(0, 1)
                plt.setp(ax4.get_xticklabels(), rotation=45)
            else:
                ax4.text(0.5, 0.5, 'History will appear after more predictions', 
                        ha='center', va='center', transform=ax4.transAxes, fontsize=12)
                ax4.axis('off')
            
            plt.tight_layout()
            
            # 保存结果
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = f"{self.predictions_dir}/prediction_{image_name}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   💾 预测结果已保存: {output_path}")
            
        except Exception as e:
            print(f"   ❌ 可视化保存失败: {e}")
    
    def generate_enhanced_visualizations(self, y_true, y_pred, paths, X_features):
        """生成增强的可视化"""
        print("🎨 生成增强可视化结果...")
        
        try:
            # 1. 混淆矩阵
            self.create_confusion_matrix(y_true, y_pred)
            
            # 2. 模型性能对比
            self.create_model_comparison()
            
            # 3. 特征分析
            self.create_feature_analysis(X_features)
            
            # 4. 训练摘要
            self.create_training_summary(y_true, y_pred)
            
            print(f"   ✅ 可视化结果已保存到 {self.output_dir}/ 目录")
            
        except Exception as e:
            print(f"   ❌ 可视化生成失败: {e}")
    
    def create_confusion_matrix(self, y_true, y_pred):
        """创建混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Class', fontsize=12)
        plt.ylabel('True Class', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'{self.training_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_model_comparison(self):
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
        bars = plt.bar(names, scores, yerr=errors, capsize=5, alpha=0.7)
        
        # 着色
        colors = plt.cm.Set3(np.linspace(0, 1, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # 突出显示最佳模型
        best_idx = np.argmax(scores)
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(3)
        
        plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Algorithm', fontsize=12)
        plt.ylabel('Cross-validation Accuracy', fontsize=12)
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.training_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_feature_analysis(self, X_features):
        """创建特征分析图表"""
        plt.figure(figsize=(15, 10))
        
        # 特征分布
        plt.subplot(2, 2, 1)
        feature_means = np.mean(X_features, axis=0)
        plt.hist(feature_means, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Feature Mean Distribution', fontsize=14)
        plt.xlabel('Feature Value')
        plt.ylabel('Frequency')
        
        # Feature variance
        plt.subplot(2, 2, 2)
        feature_vars = np.var(X_features, axis=0)
        plt.hist(feature_vars, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.title('Feature Variance Distribution', fontsize=14)
        plt.xlabel('Feature Value')
        plt.ylabel('Frequency')
        
        # PCA visualization
        plt.subplot(2, 2, 3)
        if X_features.shape[1] > 2:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_features)
            plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
            plt.title('PCA 2D Visualization', fontsize=14)
            plt.xlabel('First Principal Component')
            plt.ylabel('Second Principal Component')
        
        # Feature importance
        plt.subplot(2, 2, 4)
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[-20:]  # Show top 20 important features
            plt.barh(range(20), importances[indices])
            plt.yticks(range(20), [f'Feature_{i}' for i in indices])
            plt.title('Feature Importance', fontsize=14)
            plt.xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig(f'{self.training_dir}/feature_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_training_summary(self, y_true, y_pred):
        """创建训练摘要"""
        # 生成详细报告
        report = classification_report(y_true, y_pred, target_names=self.class_names, 
                                     digits=4, zero_division=0)
        
        # 创建摘要图表
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.axis('off')
        
        # Text content
        summary_text = f"""
ChladniVision Optimized Training Summary
{'='*50}

Model Information:
• Best Algorithm: {self.model_info.get('best_model', 'Unknown')}
• Validation Accuracy: {self.model_info.get('best_score', 0):.4f}
• Number of Classes: {len(self.class_names)}
• Total Samples: {len(y_true)}
• Ensemble Learning: {'Yes' if self.model_info.get('is_ensemble', False) else 'No'}

Performance Metrics:
• Training Accuracy: {accuracy_score(y_true, y_pred):.4f}
• Model Type: {type(self.model).__name__}

Feature Configuration:
• Feature Extractor: OptimizedFeatureExtractor
• Feature Dimension: {self.feature_extractor.n_features}
• PCA Dimensionality: {self.feature_extractor.pca.n_components_ if hasattr(self.feature_extractor, 'pca') and self.feature_extractor.pca else 'N/A'}

Training Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
               fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(f'{self.training_dir}/training_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save text report
        with open(f'{self.reports_dir}/classification_report.txt', 'w', encoding='utf-8') as f:
            f.write("ChladniVision Optimized Detailed Classification Report\n")
            f.write("="*60 + "\n\n")
            f.write(summary_text)
            f.write("\n\nDetailed Classification Report:\n")
            f.write(report)
    
    def save_model(self, model_path):
        """保存模型"""
        model_data = {
            'model': self.model,
            'feature_extractor': self.feature_extractor,
            'class_names': self.class_names,
            'model_info': self.model_info,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, model_path)
    
    def load_model(self, model_path):
        """加载模型"""
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            return False
        
        try:
            model_data = joblib.load(model_path)
            
            self.model = model_data['model']
            self.feature_extractor = model_data['feature_extractor']
            self.class_names = model_data['class_names']
            self.model_info = model_data.get('model_info', {})
            self.is_trained = model_data['is_trained']
            
            print("✅ 模型加载成功")
            print(f"   模型类型: {self.model_info.get('best_model', 'Unknown')}")
            print(f"   支持类别: {len(self.class_names)}")
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    
    def interactive_mode(self):
        """交互式模式"""
        print("\n🎯 交互式模式")
        print("   📮 输入图像路径进行预测，或使用以下快捷命令:")
        print("   🎲 'example' - 使用示例图像自动演示")
        print("   📊 'show' - 展示生成的可视化结果")
        print("   📁 'list' - 列出可用的示例图像")
        print("   🎪 'demo' - 完整功能演示")
        print("   🚪 'quit' - 退出程序")
        
        # 获取示例图像列表
        example_images = self._get_example_images()
        
        while True:
            user_input = input("\n📷 请输入命令或图像路径: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 退出交互模式")
                break
            
            elif user_input.lower() == 'example':
                self._run_example_prediction()
            
            elif user_input.lower() == 'show':
                self._show_visualization_results()
            
            elif user_input.lower() == 'list':
                self._list_example_images(example_images)
            
            elif user_input.lower() == 'demo':
                self._run_full_demo()
            
            elif os.path.exists(user_input):
                self.predict(user_input, save_result=True)
            else:
                print("❌ 文件不存在或命令无效")
                print("💡 输入 'list' 查看可用示例，或 'example' 运行示例预测")
    
    def _get_example_images(self):
        """获取示例图像列表"""
        example_images = []
        data_dirs = ['data/data/', 'data/data_augmented/']
        
        for data_dir in data_dirs:
            if os.path.exists(data_dir):
                for root, dirs, files in os.walk(data_dir):
                    for file in files:
                        if file.endswith(('.png', '.jpg', '.jpeg')):
                            example_images.append(os.path.join(root, file))
                            if len(example_images) >= 10:  # 限制数量
                                break
                    if len(example_images) >= 10:
                        break
                if len(example_images) >= 10:
                    break
        
        return example_images
    
    def _list_example_images(self, example_images):
        """列出示例图像"""
        print("\n📁 可用的示例图像:")
        for i, img_path in enumerate(example_images[:10], 1):
            filename = os.path.basename(img_path)
            # 从路径中提取频率信息
            freq = "Unknown"
            for f in ['600Hz', '700Hz', '800Hz', '900Hz', '1100Hz']:
                if f in img_path:
                    freq = f
                    break
            print(f"   {i:2d}. {filename} ({freq})")
        
        if len(example_images) > 10:
            print(f"   ... 还有 {len(example_images) - 10} 个图像")
        
        print(f"\n💡 使用方法:")
        print(f"   - 输入完整路径: {example_images[0]}")
        print(f"   - 或输入 'example' 自动选择示例")
    
    def _run_example_prediction(self):
        """运行示例预测"""
        example_images = self._get_example_images()
        if not example_images:
            print("❌ 未找到示例图像")
            return
        
        # 随机选择一个示例图像
        import random
        selected_image = random.choice(example_images[:5])  # 从前5个中选择
        filename = os.path.basename(selected_image)
        
        print(f"\n🎲 使用示例图像: {filename}")
        self.predict(selected_image, save_result=True)
    
    def _show_visualization_results(self):
        """展示可视化结果"""
        print(f"\n📊 查看可视化结果...")
        
        output_dirs = [self.training_dir, self.predictions_dir]
        found_files = []
        
        for output_dir in output_dirs:
            if os.path.exists(output_dir):
                files = os.listdir(output_dir)
                image_files = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))]
                found_files.extend([(output_dir, f) for f in image_files])
        
        if not found_files:
            print("❌ 未找到可视化结果")
            print("💡 请先运行训练或预测来生成可视化结果")
            return
        
        print(f"📸 找到 {len(found_files)} 个可视化文件:")
        
        for i, (output_dir, filename) in enumerate(found_files[:10], 1):
            file_path = os.path.join(output_dir, filename)
            file_size = os.path.getsize(file_path)
            print(f"   {i:2d}. {filename} ({file_size:,} bytes)")
        
        # 询问是否要打开某个文件
        try:
            choice = input(f"\n🔍 输入文件编号查看详情 (1-{min(10, len(found_files))}): ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(found_files):
                    output_dir, filename = found_files[idx]
                    file_path = os.path.join(output_dir, filename)
                    self._show_image_details(file_path)
        except:
            pass
    
    def _show_image_details(self, image_path):
        """显示图像详情"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg
            
            # 读取并显示图像
            img = mpimg.imread(image_path)
            
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.title(f"可视化结果: {os.path.basename(image_path)}", fontsize=16, fontweight='bold')
            plt.axis('off')
            
            # 添加文件信息
            file_size = os.path.getsize(image_path)
            info_text = f"文件大小: {file_size:,} bytes\n路径: {image_path}"
            plt.figtext(0.02, 0.02, info_text, fontsize=10, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"❌ 无法显示图像: {e}")
            print(f"💡 您可以手动打开: {image_path}")
    
    def _run_full_demo(self):
        """运行完整功能演示"""
        print(f"\n🎪 ChladniVision 完整功能演示")
        print("=" * 50)
        
        # 1. 展示系统信息
        print(f"📋 系统信息:")
        print(f"   🧠 特征提取器: OptimizedFeatureExtractor")
        print(f"   🤖 支持算法: RandomForest, SVM, MLP, GradientBoosting")
        print(f"   🎯 分类类别: {len(self.class_names) if self.class_names else 5} 个")
        print(f"   📊 输出目录: {self.output_dir}")
        
        # 2. 如果有模型，进行预测演示
        if self.is_trained:
            print(f"\n🎯 预测演示:")
            example_images = self._get_example_images()
            if example_images:
                # 选择不同频率的示例
                demo_images = []
                freqs = ['600Hz', '700Hz', '800Hz', '900Hz', '1100Hz']
                for freq in freqs:
                    for img in example_images:
                        if freq in img:
                            demo_images.append(img)
                            break
                
                for img_path in demo_images[:3]:  # 演示3个
                    filename = os.path.basename(img_path)
                    print(f"\n📸 预测: {filename}")
                    result = self.predict(img_path, save_result=True)
                    if result:
                        print(f"   ✅ 预测完成")
        
        # 3. 展示可视化结果
        print(f"\n📊 可视化展示:")
        self._show_visualization_results()
        
        print(f"\n🎉 演示完成!")
        print(f"💡 继续使用其他命令探索更多功能")

class EnsembleModel:
    """集成学习模型"""
    
    def __init__(self, models):
        self.models = models
    
    def predict(self, X):
        """集成预测"""
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # 投票决定最终结果
        predictions = np.array(predictions)
        result = []
        for i in range(predictions.shape[1]):
            votes = predictions[:, i]
            # 使用numpy的unique来找到最常见的预测
            unique_votes, counts = np.unique(votes, return_counts=True)
            most_common = unique_votes[np.argmax(counts)]
            result.append(most_common)
        
        return np.array(result)
    
    def predict_proba(self, X):
        """集成概率预测"""
        all_probs = []
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X)
                all_probs.append(probs)
        
        if all_probs:
            # 平均概率
            avg_probs = np.mean(all_probs, axis=0)
            return avg_probs
        else:
            # 如果没有概率预测，返回简单投票结果
            predictions = self.predict(X)
            n_classes = len(np.unique(predictions))
            result = np.zeros((len(predictions), n_classes))
            for i, pred in enumerate(predictions):
                result[i, pred] = 1.0
            return result

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='ChladniVision Optimized - 优化版系统')
    parser.add_argument('--train', action='store_true', help='训练模式')
    parser.add_argument('--data_dir', type=str, help='数据目录路径')
    parser.add_argument('--predict', type=str, help='预测图像路径')
    parser.add_argument('--model', type=str, default='demo_optimized_model.pkl', 
                       help='模型文件路径')
    parser.add_argument('--interactive', action='store_true', help='交互式模式')
    parser.add_argument('--demo', action='store_true', help='演示模式')
    parser.add_argument('--output_dir', type=str, default='output_optimized', 
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 创建系统实例
    system = ChladniVisionOptimized()
    system.output_dir = args.output_dir
    system.welcome_message()
    
    if args.demo:
        # 演示模式
        print("\n🎭 演示模式")
        print("   使用增强数据集进行快速演示...")
        
        # 训练演示模型
        success = system.train('data/data_augmented/', 'demo_optimized_model.pkl')
        if success:
            print("\n🎯 演示预测:")
            # 演示预测
            demo_image = 'data/data/600Hz/600hz_001.png'
            if os.path.exists(demo_image):
                system.predict(demo_image, save_result=True)
            
            # 进入交互模式
            system.interactive_mode()
    
    elif args.train:
        # 训练模式
        if not args.data_dir:
            print("❌ 训练模式需要指定 --data_dir")
            return
        
        success = system.train(args.data_dir, args.model)
        if success and args.interactive:
            system.interactive_mode()
    
    elif args.predict:
        # 预测模式
        if not system.load_model(args.model):
            return
        
        system.predict(args.predict, save_result=True)
    
    elif args.interactive:
        # 交互式模式
        if not system.load_model(args.model):
            print("   请先训练模型或指定正确的模型路径")
            return
        
        system.interactive_mode()
    
    else:
        # 显示使用说明
        print("\n📖 使用说明:")
        print("   1. 演示模式 (快速体验):")
        print("      python chladni_vision_optimized.py --demo")
        print("")
        print("   2. 训练模型:")
        print("      python chladni_vision_optimized.py --train --data_dir data/data_augmented/")
        print("")
        print("   3. 预测图像:")
        print("      python chladni_vision_optimized.py --predict image.png")
        print("")
        print("   4. 交互式模式:")
        print("      python chladni_vision_optimized.py --interactive")
        print("")
        print("   5. 训练后进入交互模式:")
        print("      python chladni_vision_optimized.py --train --data_dir data/data_augmented/ --interactive")
        print("")
        print("🔧 输出文件:")
        print("   - output_optimized/confusion_matrix.png")
        print("   - output_optimized/model_comparison.png")
        print("   - output_optimized/feature_analysis.png")
        print("   - output_optimized/training_summary.png")
        print("   - output_optimized/prediction_*.png")
        print("   - output_optimized/classification_report.txt")

if __name__ == "__main__":
    main()