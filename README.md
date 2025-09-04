# ChladniVision - 克拉尼图形智能分类系统

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Advanced-green.svg)](https://scikit-learn.org/)
[![Computer Vision](https://img.shields.io/badge/Computer%20Vision-OpenCV-red.svg)](https://opencv.org/)

**ChladniVision** 是一个专业的克拉尼图形智能分类系统，采用多模态特征提取和集成学习方法，实现高精度的声学振动模式识别。系统支持统一命令界面和传统命令模式，提供专业级的数据分析和可视化功能。

## 🎯 项目概述

克拉尼图形是由声波振动在薄板上形成的几何图案，不同频率产生不同的图案。本项目利用计算机视觉和机器学习技术，实现了克拉尼图形的自动频率识别。

### 🌟 核心特性

- **多模态特征提取**: 融合10种高级特征提取方法
- **智能算法选择**: 自动选择最优机器学习算法
- **集成学习**: 多模型融合提高准确率
- **专业可视化**: 出版级别的数据分析图表
- **统一命令系统**: 简化的操作界面

## 📊 性能指标

| 指标 | 数值 | 说明 |
|------|------|------|
| **分类准确率** | 99.94% | 测试样本表现 |
| **特征维度** | 520维 | 多模态特征融合 |
| **预测时间** | < 1秒 | 单张图片处理 |
| **支持格式** | PNG/JPG/JPEG | 常见图像格式 |

## 🚀 快速开始

### 环境要求

```bash
Python >= 3.7
pip install opencv-python numpy matplotlib seaborn scikit-learn tqdm joblib
```

### 安装与运行

```bash
# 下载项目
git clone https://github.com/JasonRobertDestiny/ChladniVision-Pro.git
cd ChladniVision-Pro

# 安装依赖
pip install -r requirements.txt

# 快速演示 (推荐)
python run_chladni.py --demo

# 统一命令系统
python run_chladni.py /analyze      # 系统分析
python run_chladni.py /build        # 构建建议
python run_chladni.py /scan         # 安全扫描
python run_chladni.py /troubleshoot  # 故障排除

# 图片分类
python run_chladni.py --predict data/data_augmented/1100Hz/1100hz_001.png --model demo_optimized_model.pkl

# 交互式模式
python run_chladni.py --interactive --model demo_optimized_model.pkl
```

## 📁 项目结构

```
ChladniVision-Pro/
├── run_chladni.py                    # 统一启动器
├── chladni_vision_unified.py         # 统一命令系统
├── chladni_vision_optimized.py       # 优化核心系统
├── src/                              # 源代码目录
│   ├── chladni_vision_pro.py         # 原始版本
│   └── feature_extractor_optimized.py # 特征提取器
├── data/                             # 数据集
│   ├── data/                        # 小数据集
│   └── data_augmented/              # 增强数据集
├── output/                          # 可视化输出
├── demo_optimized_model.pkl         # 预训练模型
├── requirements.txt                  # Python依赖
└── README.md                         # 项目说明
```

## 🔧 核心功能

### 1. 统一命令系统

- **`/analyze`** - 系统架构分析与设计优化
- **`/build`** - 用户体验优化与界面设计建议
- **`/scan`** - 安全评估与漏洞分析
- **`/troubleshoot`** - 性能分析与问题诊断

### 2. 智能训练系统

- **自动数据检测**: 智能识别数据集结构
- **多算法对比**: RandomForest, SVM, MLP, GradientBoosting
- **超参数优化**: 网格搜索自动调优
- **交叉验证**: 5折交叉验证确保稳定性

### 3. 精准预测引擎

- **多模态特征**: 10种特征提取方法融合
- **实时预测**: <1秒快速响应
- **置信度评估**: 提供预测可信度
- **概率分布**: 显示所有类别的预测概率

### 4. 交互式体验

- **自动示例**: 无需手动输入路径
- **智能提示**: 交互模式提供快捷命令
- **批量处理**: 支持多图像同时处理
- **可视化**: 实时生成专业图表

## 🧠 技术架构

### 特征提取模块

```python
特征提取流程:
├── 图像预处理 (CLAHE增强 + 标准化)
├── SIFT特征 (尺度不变特征变换)
├── LBP特征 (局部二值模式)
├── Gabor特征 (多尺度多方向)
├── Haralick特征 (纹理分析)
├── HOG特征 (方向梯度直方图)
├── 梯度特征 (Sobel算子)
├── 统计特征 (高阶统计量)
├── 频域特征 (傅里叶变换)
└── 颜色特征 (颜色直方图)
```

### 机器学习算法

| 算法 | 优势 | 适用场景 |
|------|------|----------|
| **RandomForest** | 抗过拟合能力强 | 稳定性要求高 |
| **SVM** | 高维空间分类优秀 | 小样本高精度 |
| **MLP** | 非线性建模能力强 | 复杂模式识别 |
| **GradientBoosting** | 集成学习 | 复杂数据集 |

## 📖 使用指南

### 基本使用

```bash
# 1. 快速演示 (自动训练和预测)
python run_chladni.py --demo

# 2. 训练自定义模型
python run_chladni.py --train --data_dir data/data_augmented/ --model my_model.pkl

# 3. 预测单张图片
python run_chladni.py --predict image.png --model demo_optimized_model.pkl

# 4. 交互式模式 (推荐)
python run_chladni.py --interactive --model demo_optimized_model.pkl
```

### 交互式命令

在交互模式中，您可以使用以下命令：

- `example` - 使用示例图像自动演示
- `show` - 展示生成的可视化结果
- `list` - 列出可用的示例图像
- `demo` - 完整功能演示
- `quit` - 退出程序
- 直接输入图片路径进行预测

## 📊 输出文件说明

所有生成的可视化文件保存在 `output/` 目录：

| 文件名 | 内容描述 |
|--------|----------|
| `confusion_matrix.png` | 混淆矩阵 |
| `model_comparison.png` | 算法性能对比 |
| `feature_analysis.png` | 特征分析 |
| `prediction_*.png` | 预测结果可视化 |
| `training_summary.png` | 训练过程摘要 |

## 🔍 数据集说明

### 增强数据集 (data_augmented/)

- **图片数量**: 55张
- **频率类别**: 600Hz, 700Hz, 800Hz, 900Hz, 1100Hz
- **每类数量**: 11张
- **训练时间**: ~20秒
- **适用场景**: 正常训练、演示使用

## 🛠️ 故障排除

### 常见问题

1. **模型加载失败**
   ```bash
   # 确保模型文件存在
   ls -la *.pkl
   # 使用正确的模型文件名
   python run_chladni.py --predict image.png --model demo_optimized_model.pkl
   ```

2. **图片格式不支持**
   ```bash
   # 检查支持的格式: PNG, JPG, JPEG
   ```

3. **依赖包缺失**
   ```bash
   pip install -r requirements.txt
   ```

## 🤝 贡献指南

我们欢迎各种形式的贡献！

### 开发环境设置

```bash
# 克隆仓库
git clone https://github.com/JasonRobertDestiny/ChladniVision-Pro.git
cd ChladniVision-Pro

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

## 📄 许可证

本项目采用 [MIT 许可证](LICENSE) - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

感谢以下开源项目的支持：
- [OpenCV](https://opencv.org/) - 计算机视觉库
- [Scikit-learn](https://scikit-learn.org/) - 机器学习库
- [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/) - 数据可视化
- [NumPy](https://numpy.org/) - 数值计算
- [Tqdm](https://tqdm.github.io/) - 进度条库

## 📞 联系方式

- **作者**: Jason Robert Destiny
- **邮箱**: johnrobertdestiny@gmail.com
- **GitHub**: https://github.com/JasonRobertDestiny
- **项目地址**: https://github.com/JasonRobertDestiny/ChladniVision

## 📈 项目路线图

- [ ] 支持实时摄像头输入
- [ ] 添加Web界面
- [ ] 实现深度学习模型
- [ ] 支持更多频率类别
- [ ] 移动端应用开发
- [ ] 云端部署版本

---

⭐ 如果这个项目对您有帮助，请考虑给个星标！

[![Star History Chart](https://api.star-history.com/svg?repos=JasonRobertDestiny/ChladniVision&type=Date)](https://star-history.com/#JasonRobertDestiny/ChladniVision&Date)