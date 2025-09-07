#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChladniVision 统一启动器
支持多种persona和命令模式
"""

import os
import sys
import argparse
from pathlib import Path

def show_banner():
    """显示启动横幅"""
    print("=" * 80)
    print("🚀 ChladniVision - 统一启动系统")
    print("   专业级克拉尼图形分类平台")
    print("=" * 80)
    print()

def show_usage():
    """显示使用说明"""
    print("📖 可用模式:")
    print()
    print("🎯 主要功能:")
    print("   python run_chladni.py --demo                    # 快速演示")
    print("   python run_chladni.py --train --data_dir data/data_augmented/   # 训练模型")
    print("   python run_chladni.py --predict image.png        # 预测图像")
    print("   python run_chladni.py --interactive              # 交互模式")
    print()
    print("� 快捷命令 (实验性):")
    print("   python run_chladni.py /scan       # 系统演示")
    print("   python run_chladni.py /analyze    # 交互分析")
    print("   python run_chladni.py /build      # 训练模型")
    print("   python run_chladni.py /troubleshoot # 问题诊断")
    print()
    print("🌟 常用示例:")
    print("   # 训练模型（使用增强数据集）")
    print("   python run_chladni.py --train --data_dir data/data_augmented/")
    print()
    print("   # 预测图像")
    print("   python run_chladni.py --predict data/600Hz/600hz_001.png")
    print()
    print("   # 快速演示")
    print("   python run_chladni.py --demo")
    print()
    print("   # 交互模式")
    print("   python run_chladni.py --interactive")
    print()

def main():
    """主函数"""
    show_banner()
    
    # 获取命令行参数
    args = sys.argv[1:]
    
    if not args:
        show_usage()
        return
    
    # 检查是否使用统一命令格式
    if len(args) > 0 and args[0].startswith('/'):
        # 统一命令模式已整合到优化版本
        print("ℹ️  统一命令模式已整合，使用优化版本处理...")
        print(f"   处理命令: {args[0]}")
        print()
        
        # 转换为优化版本可识别的参数
        if args[0] == '/scan':
            args = ['--demo']  # 演示功能
        elif args[0] == '/analyze':
            args = ['--interactive']  # 交互分析
        elif args[0] == '/build':
            args = ['--train', '--data_dir', 'data/data_augmented/']  # 构建模型
        elif args[0] == '/troubleshoot':
            args = ['--predict', 'data/600Hz/600hz_001.png']  # 问题诊断
        else:
            args = ['--demo']  # 默认演示
    
    # 使用优化版本处理所有命令
    try:
        # 导入优化版本
        sys.path.insert(0, os.path.dirname(__file__))
        from chladni_vision_optimized import main as optimized_main
        
        # 传递参数给优化版本
        sys.argv = ['chladni_vision_optimized.py'] + args
        optimized_main()
        
    except ImportError:
        print("❌ 无法导入优化版本")
        print("   请确保 chladni_vision_optimized.py 文件存在")
        print("   请检查依赖项是否正确安装")
        show_usage()
    except Exception as e:
        print(f"❌ 优化版本执行失败: {e}")
        print("   请检查输入参数和文件路径")

if __name__ == "__main__":
    main()