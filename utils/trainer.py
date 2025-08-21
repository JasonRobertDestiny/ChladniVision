# -*- coding: utf-8 -*-
"""
模型训练模块
包含训练过程管理、回调函数、模型保存等功能
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    CSVLogger, TensorBoard
)
from tensorflow.keras.models import load_model

class ModelTrainer:
    """
    模型训练器
    """
    
    def __init__(self, model, model_name="cnn_classifier"):
        """
        初始化训练器
        
        Args:
            model: Keras模型
            model_name: 模型名称
        """
        self.model = model
        self.model_name = model_name
        self.history = None
        self.best_model_path = None
        
        # 创建结果保存目录
        self.results_dir = f"results/{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def setup_callbacks(self, 
                       monitor='val_loss',
                       patience=10,
                       min_delta=0.001,
                       factor=0.5,
                       min_lr=1e-7,
                       save_best_only=True):
        """
        设置训练回调函数
        
        Args:
            monitor: 监控的指标
            patience: 早停耐心值
            min_delta: 最小改善值
            factor: 学习率衰减因子
            min_lr: 最小学习率
            save_best_only: 是否只保存最佳模型
            
        Returns:
            callbacks: 回调函数列表
        """
        callbacks = []
        
        # 模型检查点
        self.best_model_path = os.path.join(self.results_dir, f"{self.model_name}_best.h5")
        checkpoint = ModelCheckpoint(
            filepath=self.best_model_path,
            monitor=monitor,
            save_best_only=save_best_only,
            save_weights_only=False,
            mode='min' if 'loss' in monitor else 'max',
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # 早停
        early_stopping = EarlyStopping(
            monitor=monitor,
            patience=patience,
            min_delta=min_delta,
            mode='min' if 'loss' in monitor else 'max',
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # 学习率衰减
        lr_scheduler = ReduceLROnPlateau(
            monitor=monitor,
            factor=factor,
            patience=patience//2,
            min_lr=min_lr,
            mode='min' if 'loss' in monitor else 'max',
            verbose=1
        )
        callbacks.append(lr_scheduler)
        
        # CSV日志
        csv_logger = CSVLogger(
            os.path.join(self.results_dir, 'training_log.csv'),
            append=True
        )
        callbacks.append(csv_logger)
        
        # TensorBoard
        tensorboard = TensorBoard(
            log_dir=os.path.join(self.results_dir, 'tensorboard_logs'),
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
        callbacks.append(tensorboard)
        
        return callbacks
    
    def train(self, 
              X_train, y_train,
              X_val=None, y_val=None,
              epochs=50,
              batch_size=32,
              validation_split=0.2,
              callbacks=None,
              verbose=1):
        """
        训练模型
        
        Args:
            X_train: 训练数据
            y_train: 训练标签
            X_val: 验证数据
            y_val: 验证标签
            epochs: 训练轮数
            batch_size: 批次大小
            validation_split: 验证集比例（当X_val为None时使用）
            callbacks: 回调函数列表
            verbose: 详细程度
            
        Returns:
            history: 训练历史
        """
        print(f"开始训练模型: {self.model_name}")
        print(f"训练数据形状: {X_train.shape}")
        print(f"训练标签形状: {y_train.shape}")
        
        # 设置默认回调函数
        if callbacks is None:
            callbacks = self.setup_callbacks()
        
        # 准备验证数据
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            validation_split = None
            print(f"验证数据形状: {X_val.shape}")
        elif validation_split > 0:
            print(f"使用验证集比例: {validation_split}")
        
        # 开始训练
        start_time = datetime.now()
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        end_time = datetime.now()
        training_time = end_time - start_time
        
        print(f"\n训练完成！")
        print(f"训练时间: {training_time}")
        print(f"最佳模型保存在: {self.best_model_path}")
        
        # 保存训练历史
        self.save_training_history()
        
        return self.history
    
    def train_with_generator(self,
                           train_generator,
                           validation_generator=None,
                           epochs=50,
                           steps_per_epoch=None,
                           validation_steps=None,
                           callbacks=None,
                           verbose=1):
        """
        使用数据生成器训练模型
        
        Args:
            train_generator: 训练数据生成器
            validation_generator: 验证数据生成器
            epochs: 训练轮数
            steps_per_epoch: 每轮步数
            validation_steps: 验证步数
            callbacks: 回调函数列表
            verbose: 详细程度
            
        Returns:
            history: 训练历史
        """
        print(f"开始使用数据生成器训练模型: {self.model_name}")
        
        # 设置默认回调函数
        if callbacks is None:
            callbacks = self.setup_callbacks()
        
        # 开始训练
        start_time = datetime.now()
        
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=verbose
        )
        
        end_time = datetime.now()
        training_time = end_time - start_time
        
        print(f"\n训练完成！")
        print(f"训练时间: {training_time}")
        print(f"最佳模型保存在: {self.best_model_path}")
        
        # 保存训练历史
        self.save_training_history()
        
        return self.history
    
    def save_training_history(self):
        """
        保存训练历史
        """
        if self.history is None:
            return
        
        # 保存为JSON格式
        history_dict = {}
        for key, values in self.history.history.items():
            history_dict[key] = [float(v) for v in values]
        
        history_path = os.path.join(self.results_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        print(f"训练历史已保存: {history_path}")
    
    def plot_training_history(self, save_plot=True, show_plot=True):
        """
        绘制训练历史曲线
        
        Args:
            save_plot: 是否保存图片
            show_plot: 是否显示图片
        """
        if self.history is None:
            print("没有训练历史可以绘制")
            return
        
        history = self.history.history
        epochs = range(1, len(history['loss']) + 1)
        
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 绘制损失曲线
        ax1.plot(epochs, history['loss'], 'b-', label='训练损失', linewidth=2)
        if 'val_loss' in history:
            ax1.plot(epochs, history['val_loss'], 'r-', label='验证损失', linewidth=2)
        ax1.set_title('模型损失', fontsize=14, fontweight='bold')
        ax1.set_xlabel('轮数', fontsize=12)
        ax1.set_ylabel('损失', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 绘制准确率曲线
        if 'accuracy' in history:
            ax2.plot(epochs, history['accuracy'], 'b-', label='训练准确率', linewidth=2)
            if 'val_accuracy' in history:
                ax2.plot(epochs, history['val_accuracy'], 'r-', label='验证准确率', linewidth=2)
            ax2.set_title('模型准确率', fontsize=14, fontweight='bold')
            ax2.set_xlabel('轮数', fontsize=12)
            ax2.set_ylabel('准确率', fontsize=12)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(self.results_dir, 'training_history.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"训练历史图已保存: {plot_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def save_model(self, filepath=None, save_format='h5'):
        """
        保存模型
        
        Args:
            filepath: 保存路径
            save_format: 保存格式 ('h5' 或 'tf')
        """
        if filepath is None:
            filepath = os.path.join(self.results_dir, f"{self.model_name}_final.{save_format}")
        
        self.model.save(filepath)
        print(f"模型已保存: {filepath}")
        
        # 保存模型配置
        config_path = os.path.join(self.results_dir, 'model_config.json')
        with open(config_path, 'w') as f:
            json.dump({
                'model_name': self.model_name,
                'input_shape': self.model.input_shape[1:],
                'output_shape': self.model.output_shape[1:],
                'total_params': self.model.count_params(),
                'optimizer': self.model.optimizer.get_config(),
                'loss': self.model.loss,
                'metrics': self.model.metrics_names
            }, f, indent=2, default=str)
    
    def load_best_model(self):
        """
        加载最佳模型
        
        Returns:
            model: 加载的模型
        """
        if self.best_model_path and os.path.exists(self.best_model_path):
            self.model = load_model(self.best_model_path)
            print(f"已加载最佳模型: {self.best_model_path}")
            return self.model
        else:
            print("未找到最佳模型文件")
            return None
    
    def evaluate_model(self, X_test, y_test, verbose=1):
        """
        评估模型
        
        Args:
            X_test: 测试数据
            y_test: 测试标签
            verbose: 详细程度
            
        Returns:
            results: 评估结果
        """
        print("评估模型性能...")
        
        results = self.model.evaluate(X_test, y_test, verbose=verbose)
        
        # 创建结果字典
        result_dict = {}
        for i, metric_name in enumerate(self.model.metrics_names):
            result_dict[metric_name] = results[i]
        
        print("\n评估结果:")
        for metric, value in result_dict.items():
            print(f"{metric}: {value:.4f}")
        
        # 保存评估结果
        eval_path = os.path.join(self.results_dir, 'evaluation_results.json')
        with open(eval_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        return result_dict
    
    def predict(self, X, batch_size=32):
        """
        模型预测
        
        Args:
            X: 输入数据
            batch_size: 批次大小
            
        Returns:
            predictions: 预测结果
        """
        predictions = self.model.predict(X, batch_size=batch_size)
        return predictions
    
    def get_training_summary(self):
        """
        获取训练摘要
        
        Returns:
            summary: 训练摘要字典
        """
        if self.history is None:
            return None
        
        history = self.history.history
        
        summary = {
            'model_name': self.model_name,
            'total_epochs': len(history['loss']),
            'final_train_loss': history['loss'][-1],
            'final_train_accuracy': history.get('accuracy', [None])[-1],
            'best_val_loss': min(history.get('val_loss', [float('inf')])),
            'best_val_accuracy': max(history.get('val_accuracy', [0])),
            'results_dir': self.results_dir
        }
        
        return summary


def load_trained_model(model_path):
    """
    加载已训练的模型
    
    Args:
        model_path: 模型文件路径
        
    Returns:
        model: 加载的模型
    """
    try:
        model = load_model(model_path)
        print(f"成功加载模型: {model_path}")
        return model
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None


if __name__ == "__main__":
    print("模型训练模块已准备就绪！")
    print("主要功能：")
    print("1. 模型训练管理")
    print("2. 训练过程监控和回调")
    print("3. 模型保存和加载")
    print("4. 训练历史可视化")
    print("5. 模型评估")