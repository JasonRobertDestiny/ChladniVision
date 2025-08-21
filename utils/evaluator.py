# -*- coding: utf-8 -*-
"""
模型评估模块
包含混淆矩阵、分类报告、ROC曲线等评估功能
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import label_binarize
import json

class ModelEvaluator:
    """
    模型评估器
    """
    
    def __init__(self, class_names=None, results_dir="results"):
        """
        初始化评估器
        
        Args:
            class_names: 类别名称列表
            results_dir: 结果保存目录
        """
        self.class_names = class_names
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def evaluate_classification(self, y_true, y_pred, y_pred_proba=None, 
                              save_results=True, show_plots=True):
        """
        综合分类评估
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_pred_proba: 预测概率
            save_results: 是否保存结果
            show_plots: 是否显示图表
            
        Returns:
            results: 评估结果字典
        """
        print("开始模型评估...")
        
        results = {}
        
        # 基本指标
        results['accuracy'] = accuracy_score(y_true, y_pred)
        results['precision_macro'] = precision_score(y_true, y_pred, average='macro')
        results['recall_macro'] = recall_score(y_true, y_pred, average='macro')
        results['f1_macro'] = f1_score(y_true, y_pred, average='macro')
        results['precision_weighted'] = precision_score(y_true, y_pred, average='weighted')
        results['recall_weighted'] = recall_score(y_true, y_pred, average='weighted')
        results['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
        
        print(f"准确率: {results['accuracy']:.4f}")
        print(f"宏平均精确率: {results['precision_macro']:.4f}")
        print(f"宏平均召回率: {results['recall_macro']:.4f}")
        print(f"宏平均F1分数: {results['f1_macro']:.4f}")
        
        # 分类报告
        class_report = classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            output_dict=True
        )
        results['classification_report'] = class_report
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        results['confusion_matrix'] = cm.tolist()
        
        # 绘制混淆矩阵
        self.plot_confusion_matrix(cm, save_plot=save_results, show_plot=show_plots)
        
        # 绘制分类报告热图
        self.plot_classification_report(class_report, save_plot=save_results, show_plot=show_plots)
        
        # ROC曲线和AUC（多分类）
        if y_pred_proba is not None:
            roc_results = self.plot_roc_curves(y_true, y_pred_proba, 
                                             save_plot=save_results, show_plot=show_plots)
            results.update(roc_results)
            
            # PR曲线
            pr_results = self.plot_precision_recall_curves(y_true, y_pred_proba,
                                                         save_plot=save_results, show_plot=show_plots)
            results.update(pr_results)
        
        # 保存结果
        if save_results:
            self.save_evaluation_results(results)
        
        return results
    
    def plot_confusion_matrix(self, cm, normalize=False, save_plot=True, show_plot=True):
        """
        绘制混淆矩阵
        
        Args:
            cm: 混淆矩阵
            normalize: 是否标准化
            save_plot: 是否保存图片
            show_plot: 是否显示图片
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = '标准化混淆矩阵'
            fmt = '.2f'
        else:
            title = '混淆矩阵'
            fmt = 'd'
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=self.class_names or range(cm.shape[1]),
                   yticklabels=self.class_names or range(cm.shape[0]))
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('预测标签', fontsize=12)
        plt.ylabel('真实标签', fontsize=12)
        plt.tight_layout()
        
        if save_plot:
            filename = 'confusion_matrix_normalized.png' if normalize else 'confusion_matrix.png'
            plt.savefig(os.path.join(self.results_dir, filename), dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_classification_report(self, class_report, save_plot=True, show_plot=True):
        """
        绘制分类报告热图
        
        Args:
            class_report: 分类报告字典
            save_plot: 是否保存图片
            show_plot: 是否显示图片
        """
        # 提取数值数据
        metrics = ['precision', 'recall', 'f1-score']
        classes = [key for key in class_report.keys() 
                  if key not in ['accuracy', 'macro avg', 'weighted avg']]
        
        # 创建数据矩阵
        data = []
        labels = []
        for cls in classes:
            if isinstance(class_report[cls], dict):
                row = [class_report[cls][metric] for metric in metrics]
                data.append(row)
                labels.append(cls)
        
        if not data:
            return
        
        data = np.array(data)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlBu_r',
                   xticklabels=metrics, yticklabels=labels)
        plt.title('分类报告热图', fontsize=16, fontweight='bold')
        plt.xlabel('评估指标', fontsize=12)
        plt.ylabel('类别', fontsize=12)
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(os.path.join(self.results_dir, 'classification_report_heatmap.png'), 
                       dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_roc_curves(self, y_true, y_pred_proba, save_plot=True, show_plot=True):
        """
        绘制ROC曲线
        
        Args:
            y_true: 真实标签
            y_pred_proba: 预测概率
            save_plot: 是否保存图片
            show_plot: 是否显示图片
            
        Returns:
            roc_results: ROC结果字典
        """
        n_classes = y_pred_proba.shape[1]
        
        # 二值化标签
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        if n_classes == 2:
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
        
        # 计算每个类别的ROC曲线和AUC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # 计算微平均ROC曲线和AUC
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # 绘制ROC曲线
        plt.figure(figsize=(10, 8))
        
        # 绘制微平均ROC曲线
        plt.plot(fpr["micro"], tpr["micro"],
                label=f'微平均 ROC (AUC = {roc_auc["micro"]:.2f})',
                color='deeppink', linestyle=':', linewidth=4)
        
        # 绘制每个类别的ROC曲线
        colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
        for i, color in zip(range(n_classes), colors):
            class_name = self.class_names[i] if self.class_names else f'类别 {i}'
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{class_name} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='随机分类器')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正率 (FPR)', fontsize=12)
        plt.ylabel('真正率 (TPR)', fontsize=12)
        plt.title('ROC曲线', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(os.path.join(self.results_dir, 'roc_curves.png'), 
                       dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        # 返回AUC结果
        roc_results = {
            'auc_micro': roc_auc["micro"],
            'auc_per_class': {f'class_{i}': roc_auc[i] for i in range(n_classes)},
            'auc_macro': np.mean([roc_auc[i] for i in range(n_classes)])
        }
        
        return roc_results
    
    def plot_precision_recall_curves(self, y_true, y_pred_proba, save_plot=True, show_plot=True):
        """
        绘制精确率-召回率曲线
        
        Args:
            y_true: 真实标签
            y_pred_proba: 预测概率
            save_plot: 是否保存图片
            show_plot: 是否显示图片
            
        Returns:
            pr_results: PR结果字典
        """
        n_classes = y_pred_proba.shape[1]
        
        # 二值化标签
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        if n_classes == 2:
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
        
        # 计算每个类别的PR曲线
        precision = dict()
        recall = dict()
        pr_auc = dict()
        
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
            pr_auc[i] = auc(recall[i], precision[i])
        
        # 绘制PR曲线
        plt.figure(figsize=(10, 8))
        
        colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
        for i, color in zip(range(n_classes), colors):
            class_name = self.class_names[i] if self.class_names else f'类别 {i}'
            plt.plot(recall[i], precision[i], color=color, lw=2,
                    label=f'{class_name} (AUC = {pr_auc[i]:.2f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('召回率 (Recall)', fontsize=12)
        plt.ylabel('精确率 (Precision)', fontsize=12)
        plt.title('精确率-召回率曲线', fontsize=16, fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(os.path.join(self.results_dir, 'precision_recall_curves.png'), 
                       dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        # 返回PR AUC结果
        pr_results = {
            'pr_auc_per_class': {f'class_{i}': pr_auc[i] for i in range(n_classes)},
            'pr_auc_macro': np.mean([pr_auc[i] for i in range(n_classes)])
        }
        
        return pr_results
    
    def plot_prediction_distribution(self, y_pred_proba, save_plot=True, show_plot=True):
        """
        绘制预测概率分布
        
        Args:
            y_pred_proba: 预测概率
            save_plot: 是否保存图片
            show_plot: 是否显示图片
        """
        n_classes = y_pred_proba.shape[1]
        
        plt.figure(figsize=(12, 8))
        
        for i in range(n_classes):
            class_name = self.class_names[i] if self.class_names else f'类别 {i}'
            plt.subplot(2, (n_classes + 1) // 2, i + 1)
            plt.hist(y_pred_proba[:, i], bins=50, alpha=0.7, density=True)
            plt.title(f'{class_name} 预测概率分布')
            plt.xlabel('预测概率')
            plt.ylabel('密度')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(os.path.join(self.results_dir, 'prediction_distribution.png'), 
                       dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def save_evaluation_results(self, results):
        """
        保存评估结果
        
        Args:
            results: 评估结果字典
        """
        # 保存为JSON格式
        results_path = os.path.join(self.results_dir, 'detailed_evaluation_results.json')
        
        # 处理numpy数组
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        # 递归转换所有numpy对象
        def recursive_convert(obj):
            if isinstance(obj, dict):
                return {k: recursive_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_convert(v) for v in obj]
            else:
                return convert_numpy(obj)
        
        converted_results = recursive_convert(results)
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(converted_results, f, indent=2, ensure_ascii=False)
        
        print(f"详细评估结果已保存: {results_path}")
        
        # 保存简化的文本报告
        report_path = os.path.join(self.results_dir, 'evaluation_summary.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("模型评估报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("基本指标:\n")
            f.write(f"准确率: {results['accuracy']:.4f}\n")
            f.write(f"宏平均精确率: {results['precision_macro']:.4f}\n")
            f.write(f"宏平均召回率: {results['recall_macro']:.4f}\n")
            f.write(f"宏平均F1分数: {results['f1_macro']:.4f}\n\n")
            
            if 'auc_macro' in results:
                f.write(f"宏平均AUC: {results['auc_macro']:.4f}\n")
            if 'pr_auc_macro' in results:
                f.write(f"宏平均PR-AUC: {results['pr_auc_macro']:.4f}\n")
        
        print(f"评估摘要已保存: {report_path}")
    
    def compare_models(self, model_results_list, model_names=None):
        """
        比较多个模型的性能
        
        Args:
            model_results_list: 模型结果列表
            model_names: 模型名称列表
        """
        if model_names is None:
            model_names = [f'模型{i+1}' for i in range(len(model_results_list))]
        
        metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        
        # 创建比较数据
        comparison_data = {}
        for metric in metrics:
            comparison_data[metric] = [results.get(metric, 0) for results in model_results_list]
        
        # 绘制比较图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            axes[i].bar(model_names, comparison_data[metric])
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_ylabel('分数')
            axes[i].tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for j, v in enumerate(comparison_data[metric]):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'model_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # 保存比较结果
        comparison_path = os.path.join(self.results_dir, 'model_comparison.json')
        with open(comparison_path, 'w', encoding='utf-8') as f:
            json.dump({
                'model_names': model_names,
                'metrics': comparison_data
            }, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    print("模型评估模块已准备就绪！")
    print("主要功能：")
    print("1. 分类性能评估")
    print("2. 混淆矩阵可视化")
    print("3. ROC曲线和AUC计算")
    print("4. 精确率-召回率曲线")
    print("5. 模型性能比较")