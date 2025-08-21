#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的图片分类功能 - 解决中文显示和预测效果问题
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
import platform
import warnings
warnings.filterwarnings('ignore')

# 优化的中文字体设置
def setup_chinese_font():
    """设置中文字体支持，解决乱码问题"""
    import matplotlib.font_manager as fm
    
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
    
    # 强制设置matplotlib参数
    plt.rcParams.update({
        'font.sans-serif': font_list,
        'axes.unicode_minus': False,
        'figure.figsize': (10, 8),
        'figure.dpi': 100,
        'font.size': 12,
        'axes.titlesize': 16,
        'figure.titlesize': 18
    })
    
    # 清除matplotlib字体缓存
    try:
        fm._rebuild()
    except:
        pass
    
    print(f"✅ 字体设置完成，当前系统: {system}")
    
    # 测试中文显示
    try:
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.text(0.5, 0.5, '测试', ha='center', va='center')
        plt.close(fig)
        print("✅ 中文字体测试通过")
    except Exception as e:
        print(f"⚠️ 中文字体测试失败: {e}")

# 初始化字体设置
setup_chinese_font()

# 导入配置
from config import config

# 添加项目路径
sys.path.append(os.path.dirname(__file__))

class SimpleImageClassifier:
    """
    简单的图片分类器
    """
    
    def __init__(self, language='en'):
        # 优化的KNN分类器参数
        self.classifier = KNeighborsClassifier(
            n_neighbors=5,  # 增加邻居数量提高稳定性
            weights='distance',  # 使用距离权重
            algorithm='auto'
        )
        self.data_dir = 'data'
        self.class_names = []
        self.is_trained = False
        self.scaler = StandardScaler()
        
        # 优化的SIFT参数
        try:
            self.sift = cv2.SIFT_create(
                nfeatures=500,  # 限制特征点数量
                contrastThreshold=0.04,  # 提高对比度阈值
                edgeThreshold=10  # 边缘阈值
            )
        except Exception as e:
            print(f"⚠️ SIFT初始化警告: {e}")
            self.sift = cv2.SIFT_create()
            
        self.use_sift_features = True  # 是否使用SIFT特征
        self.language = language  # 语言设置
        self.feature_cache = {}  # 特征缓存，提高性能
    
    def extract_sift_features(self, image, dense_step=8):
        """
        优化的Dense SIFT特征提取
        """
        try:
            # 图像预处理
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # 标准化图像大小
            gray = cv2.resize(gray, (256, 256))
            
            # 应用高斯模糊减少噪声
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # 直方图均衡化增强对比度
            gray = cv2.equalizeHist(gray)
            
            # 方法1: 使用常规SIFT特征点检测
            keypoints, descriptors = self.sift.detectAndCompute(gray, None)
            
            if descriptors is not None and len(descriptors) > 10:
                # 使用K-means聚类生成词袋模型特征
                n_clusters = min(50, len(descriptors))
                if n_clusters >= 2:
                    try:
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        kmeans.fit(descriptors)
                        
                        # 生成直方图特征
                        labels = kmeans.labels_
                        hist, _ = np.histogram(labels, bins=n_clusters, range=(0, n_clusters))
                        feature_vector = hist.astype(np.float32)
                        
                        # 归一化
                        if np.sum(feature_vector) > 0:
                            feature_vector = feature_vector / np.sum(feature_vector)
                        
                        # 补充统计特征
                        stats = [
                            np.mean(gray), np.std(gray), np.median(gray),
                            np.percentile(gray, 25), np.percentile(gray, 75)
                        ]
                        
                        return np.concatenate([feature_vector, stats])
                    except Exception as e:
                        print(f"⚠️ K-means聚类失败: {e}")
                        # 回退到简单的描述子统计
                        feature_vector = np.mean(descriptors, axis=0)
                        return feature_vector
                else:
                    # 描述子太少，使用均值
                    feature_vector = np.mean(descriptors, axis=0)
                    return feature_vector
            else:
                # SIFT特征提取失败，使用增强的像素特征
                print("⚠️ SIFT特征不足，使用增强像素特征")
                return self.extract_enhanced_pixel_features(gray)
                
        except Exception as e:
            print(f"⚠️ SIFT特征提取失败: {e}，使用增强像素特征")
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            return self.extract_enhanced_pixel_features(gray)
    
    def extract_pixel_features(self, image):
        """
        提取简单的像素特征（备用方案）
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (64, 64))
        return image.flatten()
    
    def extract_enhanced_pixel_features(self, image):
        """
        提取增强的像素特征，包含多种统计特征
        """
        # 基础像素特征
        resized = cv2.resize(image, (32, 32))
        pixel_features = resized.flatten()
        
        # 统计特征
        stats = [
            np.mean(image), np.std(image), np.median(image),
            np.min(image), np.max(image),
            np.percentile(image, 25), np.percentile(image, 75)
        ]
        
        # 梯度特征
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_stats = [
            np.mean(gradient_magnitude), np.std(gradient_magnitude)
        ]
        
        # LBP特征（简化版）
        def simple_lbp(img, radius=1):
            h, w = img.shape
            lbp = np.zeros_like(img)
            for i in range(radius, h-radius):
                for j in range(radius, w-radius):
                    center = img[i, j]
                    code = 0
                    code |= (img[i-1, j-1] >= center) << 7
                    code |= (img[i-1, j] >= center) << 6
                    code |= (img[i-1, j+1] >= center) << 5
                    code |= (img[i, j+1] >= center) << 4
                    code |= (img[i+1, j+1] >= center) << 3
                    code |= (img[i+1, j] >= center) << 2
                    code |= (img[i+1, j-1] >= center) << 1
                    code |= (img[i, j-1] >= center) << 0
                    lbp[i, j] = code
            return lbp
        
        lbp = simple_lbp(image)
        lbp_hist, _ = np.histogram(lbp.flatten(), bins=16, range=(0, 256))
        lbp_hist = lbp_hist.astype(np.float32)
        if np.sum(lbp_hist) > 0:
            lbp_hist = lbp_hist / np.sum(lbp_hist)
        
        # 合并所有特征
        all_features = np.concatenate([
            pixel_features, stats, gradient_stats, lbp_hist
        ])
        
        return all_features
        
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
            
            print(f"\n{config.get_display_text('prediction_result', self.language)}")
            print(config.get_display_text('predicted_class', self.language).format(predicted_class))
            print(config.get_display_text('confidence', self.language).format(confidence))
            
            print(f"\n{config.get_display_text('class_probabilities', self.language)}")
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
        优化的图像显示功能，解决中文乱码问题
        """
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 重新设置字体确保显示正确
                plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
                
                # 创建图像显示
                fig = plt.figure(figsize=(10, 8))
                plt.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
                
                # 设置标题（解决中文显示问题）
                if self.language == 'en':
                    title_text = f'Prediction Result: {predicted_class}\nConfidence: {confidence:.2%}'
                else:
                    title_text = f'预测结果: {predicted_class}\n置信度: {confidence:.2%}'
                
                # 强制设置字体属性
                plt.title(title_text, fontsize=16, pad=20, 
                         fontproperties='Microsoft YaHei' if platform.system() == 'Windows' else 'DejaVu Sans')
                plt.axis('off')
                
                # 添加图像信息
                info_text = f'Image: {os.path.basename(image_path)}\nSize: {image.shape[1]}x{image.shape[0]}'
                plt.figtext(0.02, 0.02, info_text, fontsize=10, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
                           fontproperties='Microsoft YaHei' if platform.system() == 'Windows' else 'DejaVu Sans')
                
                # 设置窗口标题
                if hasattr(fig.canvas, 'manager'):
                    if self.language == 'en':
                        fig.canvas.manager.set_window_title(f'Classification Result - {predicted_class}')
                    else:
                        fig.canvas.manager.set_window_title(f'分类结果 - {predicted_class}')
                
                plt.tight_layout()
                plt.show(block=False)  # 非阻塞显示
                
                # 询问是否继续
                input("\n按回车键继续...")
                plt.close()
            else:
                print("❌ 无法读取图像文件")
            
        except Exception as e:
            print(f"❌ 显示图像时出错: {str(e)}")
            print(f"   图像路径: {image_path}")
            print(f"   预测结果: {predicted_class} (置信度: {confidence:.2%})")
    
    def select_language(self):
        """选择界面语言"""
        print("\n=== Language Selection / 语言选择 ===")
        print("1. English")
        print("2. 中文")
        
        while True:
            choice = input("Please select language / 请选择语言 (1/2): ").strip()
            if choice == '1':
                self.language = 'en'
                print("✅ English selected")
                break
            elif choice == '2':
                self.language = 'zh'
                print("✅ 已选择中文")
                break
            else:
                print("❌ Invalid choice / 无效选择, please enter 1 or 2 / 请输入1或2")
    
    def run_demo(self):
        """
        优化的演示流程，提供更好的用户体验
        """
        print("\n" + "="*60)
        print("🎯 ChladniVision 优化版图像分类演示")
        print("   解决中文显示问题，提升预测准确性")
        print("="*60)
        
        # 语言选择
        self.select_language()
        
        print(f"\n=== {config.get_display_text('title', self.language)} ===")
        print(f"\n🔧 {config.get_display_text('feature_selection', self.language)}")
        print(f"   {config.get_display_text('sift_option', self.language)}")
        print(f"   {config.get_display_text('pixel_option', self.language)}")
        
        while True:
            choice = input(f"\n{config.get_display_text('select_method', self.language)}").strip()
            if choice == '1':
                print(f"\n✅ {config.get_display_text('sift_selected', self.language)}")
                self.use_sift_features = True
                break
            elif choice == '2':
                print(f"\n✅ {config.get_display_text('pixel_selected', self.language)}")
                self.use_sift_features = False
                break
            else:
                print(f"❌ {config.get_display_text('invalid_choice', self.language)}")
        
        print(f"\n{config.get_display_text('training_start', self.language)}")
        accuracy = self.train_model()
        
        if accuracy is None:
            print(f"❌ {config.get_display_text('training_failed', self.language)}")
            return
        
        print("\n" + "="*60)
        print(f"🎉 {config.get_display_text('training_complete', self.language)}")
        print(f"📊 模型准确率: {accuracy:.2%}")
        print("="*60)
        
        # 提供示例图片路径提示
        if os.path.exists(self.data_dir):
            print("\n💡 示例图片路径:")
            for class_name in self.class_names[:3]:  # 显示前3个类别
                class_path = os.path.join(self.data_dir, class_name)
                if os.path.exists(class_path):
                    files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    if files:
                        example_path = os.path.join(class_path, files[0])
                        print(f"   {class_name}: {example_path}")
        
        while True:
            print(f"\n📋 {config.get_display_text('select_operation', self.language)}")
            print("   2-查看模型信息")
            choice = input("请选择 (0-退出, 1-预测图片, 2-模型信息): ").strip()
            
            if choice == '0':
                print(f"\n👋 {config.get_display_text('goodbye', self.language)}")
                break
            elif choice == '1':
                image_path = input(f"\n{config.get_display_text('enter_path', self.language)}").strip()
                
                # 处理引号
                image_path = image_path.strip('"\'')
                
                if image_path and os.path.exists(image_path):
                    print(f"\n🔍 正在分析图像: {os.path.basename(image_path)}")
                    self.predict_image(image_path)
                else:
                    print(f"❌ {config.get_display_text('file_not_found', self.language)}")
                    print(f"   输入的路径: {image_path}")
            elif choice == '2':
                self.show_model_info()
            else:  
                print(f"❌ {config.get_display_text('invalid_choice', self.language)}")
    
    def show_model_info(self):
        """
        显示模型信息
        """
        print("\n" + "="*50)
        print("📊 模型信息")
        print("="*50)
        print(f"🔧 分类器: KNN (k={self.classifier.n_neighbors})")
        print(f"📁 数据目录: {self.data_dir}")
        print(f"🏷️  类别数量: {len(self.class_names)}")
        print(f"📋 类别列表: {', '.join(self.class_names)}")
        print(f"🎯 特征类型: {'优化SIFT特征' if self.use_sift_features else '增强像素特征'}")
        print(f"✅ 训练状态: {'已训练' if self.is_trained else '未训练'}")
        print(f"🌐 界面语言: {'中文' if self.language == 'zh' else 'English'}")
    



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