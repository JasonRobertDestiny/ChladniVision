# -*- coding: utf-8 -*-
"""
克拉尼图形KNN分类器模块
基于SIFT特征的KNN分类实现
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

class ChladniKNNClassifier:
    """
    克拉尼图形KNN分类器
    专门用于基于SIFT特征的克拉尼图形分类
    """
    
    def __init__(self, n_neighbors=5, weights='uniform', metric='euclidean'):
        """
        初始化KNN分类器
        
        Args:
            n_neighbors: 邻居数量
            weights: 权重方式 ('uniform' 或 'distance')
            metric: 距离度量方式
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        
        # 创建KNN分类器
        self.knn = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric
        )
        
        # 标签编码器
        self.label_encoder = LabelEncoder()
        
        # 训练相关属性
        self.is_trained = False
        self.class_names = None
        self.feature_dim = None
        
    def optimize_hyperparameters(self, X_train, y_train, cv=5):
        """
        使用网格搜索优化超参数
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            cv: 交叉验证折数
            
        Returns:
            best_params: 最佳参数
        """
        print("正在优化KNN超参数...")
        
        # 定义参数网格
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }
        
        # 网格搜索
        grid_search = GridSearchCV(
            KNeighborsClassifier(),
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # 更新分类器参数
        self.n_neighbors = grid_search.best_params_['n_neighbors']
        self.weights = grid_search.best_params_['weights']
        self.metric = grid_search.best_params_['metric']
        
        # 重新创建分类器
        self.knn = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric=self.metric
        )
        
        print(f"最佳参数: {grid_search.best_params_}")
        print(f"最佳交叉验证得分: {grid_search.best_score_:.4f}")
        
        return grid_search.best_params_
    
    def train(self, X_train, y_train, class_names=None, optimize=True):
        """
        训练KNN分类器
        
        Args:
            X_train: 训练特征矩阵
            y_train: 训练标签
            class_names: 类别名称列表
            optimize: 是否优化超参数
        """
        print(f"开始训练KNN分类器，训练样本数: {len(X_train)}")
        
        # 检查数据
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError("训练数据不能为空")
            
        if len(X_train) != len(y_train):
            raise ValueError("特征和标签数量不匹配")
        
        # 编码标签
        y_encoded = self.label_encoder.fit_transform(y_train)
        
        # 保存类别信息
        if class_names is not None:
            self.class_names = class_names
        else:
            self.class_names = [f"Class_{i}" for i in range(len(np.unique(y_encoded)))]
        
        self.feature_dim = X_train.shape[1]
        
        # 优化超参数
        if optimize and len(X_train) > 30:  # 只有足够的样本时才优化
            self.optimize_hyperparameters(X_train, y_encoded)
        
        # 训练分类器
        print(f"使用参数训练: n_neighbors={self.n_neighbors}, weights={self.weights}, metric={self.metric}")
        self.knn.fit(X_train, y_encoded)
        
        # 计算训练准确率
        train_pred = self.knn.predict(X_train)
        train_accuracy = accuracy_score(y_encoded, train_pred)
        
        print(f"训练完成，训练准确率: {train_accuracy:.4f}")
        
        # 显示类别分布
        class_counts = Counter(y_train)
        print("\n类别分布:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} 样本")
        
        self.is_trained = True
    
    def predict(self, X_test):
        """
        预测测试样本
        
        Args:
            X_test: 测试特征矩阵
            
        Returns:
            predictions: 预测结果
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train()方法")
        
        # 预测
        y_pred_encoded = self.knn.predict(X_test)
        
        # 解码标签
        predictions = self.label_encoder.inverse_transform(y_pred_encoded)
        
        return predictions
    
    def predict_proba(self, X_test):
        """
        预测概率
        
        Args:
            X_test: 测试特征矩阵
            
        Returns:
            probabilities: 预测概率矩阵
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train()方法")
        
        return self.knn.predict_proba(X_test)
    
    def evaluate(self, X_test, y_test, verbose=True):
        """
        评估模型性能
        
        Args:
            X_test: 测试特征
            y_test: 测试标签
            verbose: 是否打印详细结果
            
        Returns:
            results: 评估结果字典
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train()方法")
        
        # 预测
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)
        
        # 计算指标
        accuracy = accuracy_score(y_test, predictions)
        
        # 编码真实标签用于混淆矩阵
        y_test_encoded = self.label_encoder.transform(y_test)
        y_pred_encoded = self.label_encoder.transform(predictions)
        
        # 分类报告
        class_report = classification_report(
            y_test_encoded, y_pred_encoded,
            target_names=self.class_names,
            output_dict=True
        )
        
        # 混淆矩阵
        conf_matrix = confusion_matrix(y_test_encoded, y_pred_encoded)
        
        results = {
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'predictions': predictions,
            'probabilities': probabilities
        }
        
        if verbose:
            print(f"\n=== KNN分类器评估结果 ===")
            print(f"测试准确率: {accuracy:.4f}")
            print(f"\n分类报告:")
            print(classification_report(y_test_encoded, y_pred_encoded, target_names=self.class_names))
        
        return results
    
    def plot_confusion_matrix(self, conf_matrix, class_names=None, figsize=(8, 6)):
        """
        绘制混淆矩阵
        
        Args:
            conf_matrix: 混淆矩阵
            class_names: 类别名称
            figsize: 图像大小
        """
        if class_names is None:
            class_names = self.class_names
        
        plt.figure(figsize=figsize)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.tight_layout()
        plt.show()
    
    def cross_validate(self, X, y, cv=5):
        """
        交叉验证
        
        Args:
            X: 特征矩阵
            y: 标签
            cv: 交叉验证折数
            
        Returns:
            cv_scores: 交叉验证得分
        """
        # 编码标签
        y_encoded = self.label_encoder.fit_transform(y)
        
        # 交叉验证
        cv_scores = cross_val_score(self.knn, X, y_encoded, cv=cv, scoring='accuracy')
        
        print(f"\n=== {cv}折交叉验证结果 ===")
        print(f"各折得分: {cv_scores}")
        print(f"平均得分: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return cv_scores
    
    def get_feature_importance(self, X_train, y_train, feature_names=None):
        """
        分析特征重要性（基于特征方差和类别分离度）
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            feature_names: 特征名称列表
            
        Returns:
            importance_scores: 特征重要性得分
        """
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
        
        # 计算每个特征的方差
        feature_var = np.var(X_train, axis=0)
        
        # 计算类间分离度
        y_encoded = self.label_encoder.fit_transform(y_train)
        unique_classes = np.unique(y_encoded)
        
        class_separation = np.zeros(X_train.shape[1])
        
        for i in range(X_train.shape[1]):
            class_means = []
            for class_label in unique_classes:
                class_mask = y_encoded == class_label
                if np.sum(class_mask) > 0:
                    class_means.append(np.mean(X_train[class_mask, i]))
            
            if len(class_means) > 1:
                class_separation[i] = np.var(class_means)
        
        # 综合重要性得分（归一化后的方差 + 类间分离度）
        feature_var_norm = feature_var / np.max(feature_var) if np.max(feature_var) > 0 else feature_var
        class_sep_norm = class_separation / np.max(class_separation) if np.max(class_separation) > 0 else class_separation
        
        importance_scores = feature_var_norm + class_sep_norm
        
        # 创建重要性排序
        importance_ranking = sorted(zip(feature_names, importance_scores), 
                                  key=lambda x: x[1], reverse=True)
        
        print("\n=== 特征重要性分析 ===")
        print("前10个最重要的特征:")
        for i, (name, score) in enumerate(importance_ranking[:10]):
            print(f"{i+1:2d}. {name}: {score:.4f}")
        
        return importance_scores
    
    def save_model(self, filepath):
        """
        保存训练好的模型
        
        Args:
            filepath: 保存路径
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，无法保存")
        
        model_data = {
            'knn': self.knn,
            'label_encoder': self.label_encoder,
            'n_neighbors': self.n_neighbors,
            'weights': self.weights,
            'metric': self.metric,
            'class_names': self.class_names,
            'feature_dim': self.feature_dim,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"KNN模型已保存到: {filepath}")
    
    def load_model(self, filepath):
        """
        加载训练好的模型
        
        Args:
            filepath: 模型文件路径
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.knn = model_data['knn']
        self.label_encoder = model_data['label_encoder']
        self.n_neighbors = model_data['n_neighbors']
        self.weights = model_data['weights']
        self.metric = model_data['metric']
        self.class_names = model_data['class_names']
        self.feature_dim = model_data['feature_dim']
        self.is_trained = model_data['is_trained']
        
        print(f"KNN模型已从 {filepath} 加载")
        print(f"模型参数: n_neighbors={self.n_neighbors}, weights={self.weights}, metric={self.metric}")
        print(f"类别数量: {len(self.class_names)}, 特征维度: {self.feature_dim}")


if __name__ == "__main__":
    # 测试代码
    print("克拉尼图形KNN分类器测试")
    
    # 创建模拟数据
    np.random.seed(42)
    n_samples = 100
    n_features = 50
    n_classes = 3
    
    # 生成测试数据
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    y = [f"Pattern_{i}" for i in y]  # 转换为字符串标签
    
    # 分割数据
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 创建和训练分类器
    classifier = ChladniKNNClassifier()
    classifier.train(X_train, y_train, optimize=False)
    
    # 评估
    results = classifier.evaluate(X_test, y_test)
    
    print("\nKNN分类器测试完成！")