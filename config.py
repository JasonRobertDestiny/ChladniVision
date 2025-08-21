# -*- coding: utf-8 -*-
"""
ChladniVision Configuration
克拉德尼图形分类系统配置文件
"""

import matplotlib.pyplot as plt
import matplotlib
import os
import platform

class Config:
    """配置类，管理系统设置"""
    
    def __init__(self):
        self.setup_matplotlib()
        self.setup_paths()
        self.setup_model_params()
    
    def setup_matplotlib(self):
        """设置matplotlib显示配置"""
        # 检测系统并设置合适的中文字体
        system = platform.system()
        
        if system == "Windows":
            # Windows系统字体优先级
            font_list = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi']
        elif system == "Darwin":  # macOS
            font_list = ['PingFang SC', 'Hiragino Sans GB', 'STHeiti']
        else:  # Linux
            font_list = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
        
        # 添加默认字体作为后备
        font_list.extend(['DejaVu Sans', 'Arial', 'sans-serif'])
        
        # 设置matplotlib参数
        plt.rcParams['font.sans-serif'] = font_list
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.figsize'] = (10, 8)
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 12
        
        # 设置图表样式
        plt.style.use('default')
        
    def setup_paths(self):
        """设置路径配置"""
        self.data_dir = "data"
        self.models_dir = "models"
        self.results_dir = "results"
        
        # 创建必要的目录
        for dir_path in [self.models_dir, self.results_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def setup_model_params(self):
        """设置模型参数"""
        # SIFT参数
        self.sift_params = {
            'step': 8,
            'size': 16,
            'fast': True
        }
        
        # KNN参数
        self.knn_params = {
            'n_neighbors': 3,
            'weights': 'distance',
            'algorithm': 'auto'
        }
        
        # 图像处理参数
        self.image_params = {
            'target_size': (256, 256),
            'pixel_size': (64, 64),
            'sift_features': 100,
            'augmentation': {
                'rotation_angles': [90, 180, 270],
                'flip_modes': [0, 1],  # 0: 垂直翻转, 1: 水平翻转
                'scale_factors': [0.9, 1.1]
            }
        }
    
    def get_display_text(self, key, lang='en'):
        """获取多语言显示文本"""
        texts = {
            'title': {
                'en': 'ChladniVision - Chladni Figure Classification Demo',
                'zh': 'ChladniVision - 克拉德尼图形分类演示'
            },
            'feature_selection': {
                'en': 'Feature Extraction Method Selection:',
                'zh': '特征提取方法选择:'
            },
            'sift_option': {
                'en': '1. Dense SIFT Features (Recommended, More Accurate)',
                'zh': '1. Dense SIFT特征 (推荐，更准确)'
            },
            'pixel_option': {
                'en': '2. Pixel Features (Simple, Fast)',
                'zh': '2. 像素特征 (简单，速度快)'
            },
            'select_method': {
                'en': 'Please select feature extraction method (1/2): ',
                'zh': '请选择特征提取方法 (1/2): '
            },
            'sift_selected': {
                'en': '✅ Dense SIFT features selected',
                'zh': '✅ 已选择Dense SIFT特征'
            },
            'pixel_selected': {
                'en': '✅ Pixel features selected',
                'zh': '✅ 已选择像素特征'
            },
            'invalid_choice': {
                'en': '❌ Invalid choice, please enter 1 or 2',
                'zh': '❌ 无效选择，请输入1或2'
            },
            'training_start': {
                'en': '🚀 Starting model training...',
                'zh': '🚀 开始训练模型...'
            },
            'training_complete': {
                'en': '🎯 Model training completed! Starting prediction demo...',
                'zh': '🎯 模型训练完成！开始预测演示...'
            },
            'select_operation': {
                'en': 'Select operation: 1-Predict image, 0-Exit: ',
                'zh': '选择操作: 1-预测图片, 0-退出: '
            },
            'enter_path': {
                'en': 'Please enter image path: ',
                'zh': '请输入图像路径: '
            },
            'file_not_found': {
                'en': '❌ File not found, please check the path',
                'zh': '❌ 文件不存在，请检查路径'
            },
            'prediction_result': {
                'en': '🎯 Prediction Result:',
                'zh': '🎯 预测结果:'
            },
            'predicted_class': {
                'en': '📋 Predicted Class: {}',
                'zh': '📋 预测类别: {}'
            },
            'confidence': {
                'en': '🎲 Confidence: {:.2f}%',
                'zh': '🎲 置信度: {:.2f}%'
            },
            'class_probabilities': {
                'en': '📊 Class Probabilities:',
                'zh': '📊 各类别概率:'
            },
            'goodbye': {
                'en': '👋 Thank you for using ChladniVision!',
                'zh': '👋 感谢使用 ChladniVision！'
            },
            'training_failed': {
                'en': '❌ Training failed, exiting demo',
                'zh': '❌ 训练失败，退出演示'
            },
            'enhanced_title': {
                'en': 'ChladniVision Enhanced - Advanced Chladni Pattern Classifier',
                'zh': 'ChladniVision 增强版 - 高级克拉德尼图形分类器'
            },
            'data_augmentation': {
                'en': 'Data Augmentation: Enabled (8x more training samples)',
                'zh': '数据增强: 已启用 (训练样本增加8倍)'
            },
            'enhanced_sift': {
                'en': 'Enhanced Dense SIFT Features + Data Augmentation',
                'zh': '增强密集SIFT特征 + 数据增强'
            },
            'processing_images': {
                'en': 'Processing images with augmentation...',
                'zh': '正在处理图像并进行数据增强...'
            },
            'augmentation_complete': {
                'en': 'Data augmentation complete. Generated {} samples from {} original images.',
                'zh': '数据增强完成。从{}张原始图像生成了{}个样本。'
            },
            'feature_extraction_method': {
                'en': 'Feature extraction method: {}',
                'zh': '特征提取方法: {}'
            },
            'model_performance': {
                'en': 'Model Performance Summary:',
                'zh': '模型性能总结:'
            },
            'accuracy_score': {
                'en': 'Accuracy: {:.2%}',
                'zh': '准确率: {:.2%}'
            },
            'sample_count': {
                'en': 'Training samples: {} | Test samples: {}',
                'zh': '训练样本: {} | 测试样本: {}'
            }
        }
        
        return texts.get(key, {}).get(lang, texts.get(key, {}).get('en', key))

# 全局配置实例
config = Config()