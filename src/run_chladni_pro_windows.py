#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChladniVision Pro 启动器 - Windows兼容版
完全修复Unicode编码问题，提供清晰的界面
"""

import os
import sys
import subprocess
import warnings
warnings.filterwarnings('ignore')

def safe_print(text):
    """完全兼容Windows的安全打印函数"""
    try:
        # 移除所有emoji和特殊字符
        clean_text = text
        emoji_replacements = {
            '🎵': '[音乐]', '🚀': '[启动]', '📁': '[文件夹]', '❌': '[失败]', '✅': '[成功]',
            '🤖': '[AI]', '🔍': '[搜索]', '📊': '[图表]', '🎯': '[目标]', '🎲': '[概率]',
            '📈': '[上升]', '💾': '[保存]', '⏱️': '[时间]', '🎉': '[完成]', '👋': '[再见]',
            '📷': '[相机]', '📖': '[说明]', '🔧': '[设置]', '⭐': '[星级]', '🔬': '[科学]',
            '🎨': '[艺术]', '🌟': '[亮点]', '💡': '[提示]', '🎪': '[演示]', '🏆': '[奖杯]',
            '📋': '[列表]', '🔄': '[交互]', '✨': '[闪亮]', '🔔': '[通知]', '📱': '[手机]',
            '💻': '[电脑]', '🌐': '[网络]', '🎮': '[游戏]', '🎁': '[礼物]', '🎊': '[派对]'
        }
        
        for emoji, replacement in emoji_replacements.items():
            clean_text = clean_text.replace(emoji, replacement)
        
        print(clean_text)
    except Exception:
        # 如果仍有问题，使用最基础的ASCII字符
        ascii_text = text.encode('ascii', 'ignore').decode('ascii')
        print(ascii_text)

def clear_screen():
    """清屏"""
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    """主启动函数"""
    clear_screen()
    
    safe_print("=" * 70)
    safe_print("[音乐] ChladniVision Pro - 专业版启动器")
    safe_print("   增强特征提取 | 智能算法选择 | 精美可视化")
    safe_print("=" * 70)
    safe_print("")
    safe_print("请选择功能:")
    safe_print("1. [启动] 快速演示 (推荐)")
    safe_print("2. [图表] 训练专业模型")
    safe_print("3. [目标] 预测单张图片")
    safe_print("4. [交互] 交互式预测")
    safe_print("5. [文件夹] 批量预测")
    safe_print("6. [列表] 查看系统信息")
    safe_print("7. [再见] 退出")
    safe_print("")
    
    while True:
        try:
            choice = input("请输入选择 (1-7): ").strip()
            
            if choice == '1':
                quick_demo()
            elif choice == '2':
                train_model()
            elif choice == '3':
                predict_image()
            elif choice == '4':
                interactive_predict()
            elif choice == '5':
                batch_predict()
            elif choice == '6':
                show_system_info()
            elif choice == '7':
                safe_print("[再见] 感谢使用 ChladniVision Pro!")
                break
            else:
                safe_print("无效选择，请重新输入")
        except KeyboardInterrupt:
            safe_print("\n[再见] 程序已退出")
            break
        except Exception as e:
            safe_print(f"发生错误: {e}")

def quick_demo():
    """快速演示"""
    clear_screen()
    safe_print("[演示] 快速演示模式")
    safe_print("-" * 40)
    safe_print("将使用增强数据集进行快速演示...")
    safe_print("预计时间: 1-2分钟")
    safe_print("")
    
    # 检查数据目录
    if not os.path.exists("data_augmented"):
        safe_print("[失败] 未找到 data_augmented/ 目录")
        safe_print("请确保数据集文件存在")
        input("按回车键继续...")
        return
    
    # 运行演示
    try:
        safe_print("[启动] 开始演示...")
        subprocess.run([
            sys.executable, "chladni_vision_pro.py", 
            "--train", "--data_dir", "data_augmented/", 
            "--model", "demo_model.pkl"
        ], check=True, timeout=300)
        
        # 演示预测
        demo_image = "data/600Hz/600hz_001.png"
        if os.path.exists(demo_image):
            safe_print("\n[目标] 演示预测:")
            subprocess.run([
                sys.executable, "chladni_vision_pro.py", 
                "--predict", demo_image, "--model", "demo_model.pkl"
            ], check=True)
        
        safe_print("\n[完成] 演示完成!")
        
    except subprocess.TimeoutExpired:
        safe_print("[失败] 演示超时，请重试")
    except subprocess.CalledProcessError as e:
        safe_print(f"[失败] 演示失败: {e}")
    except Exception as e:
        safe_print(f"[失败] 演示过程中出现错误: {e}")
    
    input("按回车键继续...")

def train_model():
    """训练模型"""
    clear_screen()
    safe_print("[启动] 训练专业模型")
    safe_print("-" * 40)
    
    # 选择数据集
    safe_print("可用数据集:")
    safe_print("1. data/ (小数据集 - 5张图片)")
    safe_print("2. data_augmented/ (增强数据集 - 55张图片)")
    safe_print("3. extracted_frames_full/ (大数据集 - 210张图片)")
    
    try:
        data_choice = input("选择数据集 (1-3): ").strip()
        
        if data_choice == '1':
            data_dir = "data"
            model_name = "pro_small_model.pkl"
        elif data_choice == '2':
            data_dir = "data_augmented"
            model_name = "pro_enhanced_model.pkl"
        elif data_choice == '3':
            data_dir = "extracted_frames_full"
            model_name = "pro_large_model.pkl"
        else:
            safe_print("无效选择")
            input("按回车键继续...")
            return
        
        safe_print(f"[信息] 开始训练: {data_dir}")
        safe_print("[提示] 专业训练可能需要几分钟时间...")
        
        subprocess.run([
            sys.executable, "chladni_vision_pro.py", 
            "--train", "--data_dir", data_dir, 
            "--model", model_name
        ], check=True)
        
        safe_print("[完成] 训练完成!")
        
    except subprocess.CalledProcessError as e:
        safe_print(f"[失败] 训练失败: {e}")
    except Exception as e:
        safe_print(f"[失败] 训练过程中出现错误: {e}")
    
    input("按回车键继续...")

def predict_image():
    """预测图片"""
    clear_screen()
    safe_print("[目标] 图片预测")
    safe_print("-" * 40)
    
    # 选择模型
    models = []
    model_files = [
        ("pro_small_model.pkl", "小数据集模型"),
        ("pro_enhanced_model.pkl", "增强数据集模型"),
        ("pro_large_model.pkl", "大数据集模型"),
        ("demo_model.pkl", "演示模型")
    ]
    
    for i, (model_file, desc) in enumerate(model_files, 1):
        if os.path.exists(model_file):
            models.append((model_file, desc))
            safe_print(f"{i}. {model_file} ({desc})")
    
    if not models:
        safe_print("[失败] 没有找到训练好的模型，请先训练")
        input("按回车键继续...")
        return
    
    try:
        model_choice = int(input("选择模型: ").strip()) - 1
        if 0 <= model_choice < len(models):
            model_file, _ = models[model_choice]
        else:
            safe_print("无效选择")
            input("按回车键继续...")
            return
    except ValueError:
        safe_print("输入错误")
        input("按回车键继续...")
        return
    
    # 输入图片路径
    image_path = input("输入图片路径: ").strip()
    if not os.path.exists(image_path):
        safe_print("[失败] 图片文件不存在")
        input("按回车键继续...")
        return
    
    try:
        subprocess.run([
            sys.executable, "chladni_vision_pro.py", 
            "--predict", image_path, "--model", model_file
        ], check=True)
    except subprocess.CalledProcessError as e:
        safe_print(f"[失败] 预测失败: {e}")
    except Exception as e:
        safe_print(f"[失败] 预测过程中出现错误: {e}")
    
    input("按回车键继续...")

def interactive_predict():
    """交互式预测"""
    clear_screen()
    safe_print("[交互] 交互式预测")
    safe_print("-" * 40)
    
    # 选择模型
    models = []
    model_files = [
        ("pro_small_model.pkl", "小数据集模型"),
        ("pro_enhanced_model.pkl", "增强数据集模型"),
        ("pro_large_model.pkl", "大数据集模型"),
        ("demo_model.pkl", "演示模型")
    ]
    
    for i, (model_file, desc) in enumerate(model_files, 1):
        if os.path.exists(model_file):
            models.append((model_file, desc))
            safe_print(f"{i}. {model_file} ({desc})")
    
    if not models:
        safe_print("[失败] 没有找到训练好的模型，请先训练")
        input("按回车键继续...")
        return
    
    try:
        model_choice = int(input("选择模型: ").strip()) - 1
        if 0 <= model_choice < len(models):
            model_file, _ = models[model_choice]
        else:
            safe_print("无效选择")
            input("按回车键继续...")
            return
    except ValueError:
        safe_print("输入错误")
        input("按回车键继续...")
        return
    
    safe_print("[提示] 将启动交互式预测模式")
    safe_print("[提示] 在交互模式中输入 'quit' 退出")
    
    try:
        subprocess.run([
            sys.executable, "chladni_vision_pro.py", 
            "--interactive", "--model", model_file
        ], check=True)
    except subprocess.CalledProcessError as e:
        safe_print(f"[失败] 交互式预测失败: {e}")
    except Exception as e:
        safe_print(f"[失败] 交互式预测过程中出现错误: {e}")

def batch_predict():
    """批量预测"""
    clear_screen()
    safe_print("[文件夹] 批量预测")
    safe_print("-" * 40)
    
    # 选择模型
    models = []
    model_files = [
        ("pro_small_model.pkl", "小数据集模型"),
        ("pro_enhanced_model.pkl", "增强数据集模型"),
        ("pro_large_model.pkl", "大数据集模型"),
        ("demo_model.pkl", "演示模型")
    ]
    
    for i, (model_file, desc) in enumerate(model_files, 1):
        if os.path.exists(model_file):
            models.append((model_file, desc))
            safe_print(f"{i}. {model_file} ({desc})")
    
    if not models:
        safe_print("[失败] 没有找到训练好的模型，请先训练")
        input("按回车键继续...")
        return
    
    try:
        model_choice = int(input("选择模型: ").strip()) - 1
        if 0 <= model_choice < len(models):
            model_file, _ = models[model_choice]
        else:
            safe_print("无效选择")
            input("按回车键继续...")
            return
    except ValueError:
        safe_print("输入错误")
        input("按回车键继续...")
        return
    
    safe_print("[提示] 将启动交互式模式")
    safe_print("[提示] 在交互模式中输入 'batch' 进入批量预测")
    
    try:
        subprocess.run([
            sys.executable, "chladni_vision_pro.py", 
            "--interactive", "--model", model_file
        ], check=True)
    except subprocess.CalledProcessError as e:
        safe_print(f"[失败] 批量预测失败: {e}")
    except Exception as e:
        safe_print(f"[失败] 批量预测过程中出现错误: {e}")

def show_system_info():
    """显示系统信息"""
    clear_screen()
    safe_print("[列表] 系统信息")
    safe_print("-" * 40)
    safe_print("ChladniVision Pro - 专业版系统")
    safe_print("")
    safe_print("主要特性:")
    safe_print("   [成功] 多模态特征提取 (SIFT+LBP+梯度+纹理+频域)")
    safe_print("   [成功] 智能算法选择与超参数优化")
    safe_print("   [成功] 鲁棒特征缩放与降维")
    safe_print("   [成功] 实时预测与批量处理")
    safe_print("   [成功] 交互式可视化界面")
    safe_print("   [成功] 详细性能分析与报告")
    safe_print("")
    safe_print("支持的算法:")
    safe_print("   * KNN (K近邻)")
    safe_print("   * Random Forest (随机森林)")
    safe_print("   * SVM (支持向量机)")
    safe_print("   * MLP (多层感知机)")
    safe_print("   * Gradient Boosting (梯度提升)")
    safe_print("")
    safe_print("输出文件:")
    safe_print("   - output/enhanced_confusion_matrix.png")
    safe_print("   - output/model_comparison.png")
    safe_print("   - output/feature_analysis.png")
    safe_print("   - output/enhanced_prediction_*.png")
    safe_print("   - output/detailed_classification_report.txt")
    safe_print("")
    safe_print("使用提示:")
    safe_print("   * 首次使用推荐快速演示模式")
    safe_print("   * 大数据集训练需要较长时间但准确率更高")
    safe_print("   * 交互模式支持批量预测和统计查看")
    safe_print("   * 所有生成的可视化图片都保存在output/目录")
    safe_print("")
    
    input("按回车键继续...")

if __name__ == "__main__":
    main()