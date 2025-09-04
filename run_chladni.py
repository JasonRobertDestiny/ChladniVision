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
    print("🎯 统一命令模式 (推荐):")
    print("   python run_chladni.py /analyze")
    print("   python run_chladni.py /build")
    print("   python run_chladni.py /scan")
    print("   python run_chladni.py /troubleshoot")
    print()
    print("🔧 传统模式:")
    print("   python run_chladni.py --demo                    # 快速演示")
    print("   python run_chladni.py --train --data_dir data/data_augmented/   # 训练模型")
    print("   python run_chladni.py --predict image.png        # 预测图像")
    print("   python run_chladni.py --interactive              # 交互模式")
    print()
    print("💡 Persona 说明:")
    print("   /analyze    - 系统架构分析与设计优化")
    print("   /build      - 用户体验优化与界面设计")
    print("   /scan       - 安全评估与漏洞分析")
    print("   /troubleshoot - 性能分析与问题诊断")
    print()
    print("🌟 常用示例:")
    print("   # 训练模型（使用增强数据集）")
    print("   python run_chladni.py --train --data_dir data/data_augmented/")
    print()
    print("   # 预测图像")
    print("   python run_chladni.py --predict data/data_augmented/1100Hz/1100hz_001.png")
    print()
    print("   # 快速演示")
    print("   python run_chladni.py --demo")
    print()
    print("   # 统一命令")
    print("   python run_chladni.py /analyze")
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
        # 使用统一命令系统
        try:
            # 导入统一命令系统
            sys.path.insert(0, os.path.dirname(__file__))
            from chladni_vision_unified import main as unified_main
            
            # 传递参数给统一系统
            sys.argv = ['chladni_vision_unified.py'] + args
            unified_main()
            
        except ImportError:
            print("❌ 无法导入统一命令系统")
            print("   请确保 chladni_vision_unified.py 文件存在")
        except Exception as e:
            print(f"❌ 统一命令执行失败: {e}")
    
    else:
        # 使用传统优化版本
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
            print("   尝试使用原始版本...")
            
            # 回退到原始版本
            try:
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
                from chladni_vision_pro import main as original_main
                
                sys.argv = ['chladni_vision_pro.py'] + args
                original_main()
                
            except ImportError:
                print("❌ 无法导入任何版本的ChladniVision")
                print("   请检查文件结构和依赖项")
            except Exception as e:
                print(f"❌ 原始版本执行失败: {e}")
        except Exception as e:
            print(f"❌ 优化版本执行失败: {e}")

if __name__ == "__main__":
    main()