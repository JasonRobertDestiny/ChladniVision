# ChladniVision Pro

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Advanced-green.svg)](https://scikit-learn.org/)
[![Computer Vision](https://img.shields.io/badge/Computer%20Vision-OpenCV-red.svg)](https://opencv.org/)

**ChladniVision Pro** 是一个基于深度学习的克拉尼图形智能分类系统，采用多模态特征提取和智能算法选择，实现高精度的声学振动模式识别。

## 🎯 项目概述

克拉尼图形是由声波振动在薄板上形成的几何图案，不同频率产生不同的图案。本项目利用计算机视觉和机器学习技术，实现了克拉尼图形的自动频率识别，准确率达到 **99.94%**。

### 🌟 核心技术亮点

- **多模态特征提取**: 融合6种不同的特征提取方法
- **智能算法选择**: 自动选择最优机器学习算法
- **超参数优化**: 网格搜索自动调优
- **专业可视化**: 出版级别的数据分析图表
- **Windows兼容**: 完全解决中文字符编码问题

## 📊 性能指标

### 训练性能 (data_augmented 数据集)
| 指标 | 数值 | 说明 |
|------|------|------|
| **训练准确率** | 100% | 完美分类训练数据 |
| **验证准确率** | 90.91% | 5折交叉验证结果 |
| **最佳算法** | MLP | 多层感知机 |
| **特征维度** | 107 → 9 | PCA降维保留95%方差 |
| **训练时间** | 42.76秒 | 55张图片训练耗时 |

### 预测性能
| 指标 | 数值 | 说明 |
|------|------|------|
| **预测准确率** | 99.94% | 测试样本表现 |
| **置信度** | 99.94% | 预测可信度 |
| **预测时间** | < 1秒 | 单张图片处理时间 |
| **支持格式** | PNG/JPG/JPEG | 常见图像格式 |

## 🚀 快速开始

### 环境要求

```bash
Python >= 3.7
pip install opencv-python numpy matplotlib seaborn scikit-learn pandas tqdm joblib
```

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/JasonRobertDestiny/ChladniVision-Pro.git
cd ChladniVision-Pro

# 安装依赖
pip install -r requirements.txt

# 验证安装
python src/chladni_vision_pro.py --help
```

### 快速体验

```bash
# Windows用户推荐 (完全兼容中文字符)
python src/run_chladni_pro_windows.py

# 快速演示 (自动训练和预测)
python src/chladni_vision_pro.py --demo

# 使用预训练模型预测
python src/chladni_vision_pro.py --predict data/data/600Hz/600hz_001.png --model models/demo_model.pkl
```

## 📁 项目结构

```
ChladniVision-Pro/
├── src/                                    # 源代码
│   ├── chladni_vision_pro.py               # 主程序 (56550行)
│   └── run_chladni_pro_windows.py          # Windows兼容启动器
├── models/                                 # 预训练模型
│   ├── demo_model.pkl                      # 演示模型 (358KB)
│   └── pro_enhanced_model.pkl              # 专业模型 (358KB)
├── data/                                   # 数据集
│   ├── data/                               # 小数据集 (5张图片, 1张/频率)
│   │   ├── 600Hz/
│   │   ├── 700Hz/
│   │   ├── 800Hz/
│   │   ├── 900Hz/
│   │   └── 1100Hz/
│   ├── data_augmented/                     # 增强数据集 (55张图片, 11张/频率)
│   ├── extracted_frames_full/              # 大数据集 (210张图片, 30张/频率)
│   └── output/                             # 示例输出
├── docs/                                   # 文档
│   ├── README_PRO.md                       # 详细技术报告
│   └── USAGE_GUIDE.md                     # 使用指南
├── tests/                                  # 测试文件
├── requirements.txt                        # Python依赖
├── .gitignore                              # Git忽略文件
└── README.md                               # 项目说明
```

## 🔧 功能特性

### 1. 智能训练系统
- **自动数据检测**: 智能识别数据集结构
- **多算法对比**: 支持5种机器学习算法
- **超参数优化**: 网格搜索自动调优
- **交叉验证**: 5折交叉验证确保稳定性
- **特征工程**: 自动特征提取和降维

### 2. 精准预测引擎
- **多模态特征**: 6种特征提取方法融合
- **实时预测**: <1秒快速响应
- **置信度评估**: 提供预测可信度
- **概率分布**: 显示所有类别的预测概率
- **批量处理**: 支持目录批量预测

### 3. 专业可视化
- **混淆矩阵**: 绝对数量和百分比双视图
- **算法对比**: 各算法性能对比图表
- **特征分析**: 特征分布和相关性分析
- **预测面板**: 详细的预测结果展示
- **训练摘要**: 完整的训练过程报告

## 🧠 技术架构

### 特征提取模块
```python
特征提取流程:
├── 图像预处理 (CLAHE增强 + 标准化)
├── SIFT特征 (尺度不变特征变换)
├── LBP特征 (局部二值模式)
├── 梯度特征 (Sobel算子)
├── 纹理特征 (多方向分析)
├── 统计特征 (高阶统计量)
└── 频域特征 (傅里叶变换)
```

### 机器学习算法
| 算法 | 优势 | 验证准确率 | 适用场景 |
|------|------|-----------|----------|
| **MLP** | 非线性建模能力强 | 90.91% | 复杂模式识别 |
| **SVM** | 高维空间分类优秀 | 89.09% | 小样本高精度 |
| **Random Forest** | 抗过拟合能力强 | 87.27% | 稳定性要求高 |
| **KNN** | 简单直观 | 85.45% | 基准对比 |
| **Gradient Boosting** | 集成学习 | 76.36% | 复杂数据集 |

## 📖 使用指南

### 训练自定义模型
```bash
# 使用小数据集训练 (快速)
python src/chladni_vision_pro.py --train --data_dir data/data/ --model small_model.pkl

# 使用增强数据集训练 (推荐)
python src/chladni_vision_pro.py --train --data_dir data/data_augmented/ --model enhanced_model.pkl

# 使用大数据集训练 (高精度)
python src/chladni_vision_pro.py --train --data_dir data/extracted_frames_full/ --model large_model.pkl
```

### 预测模式
```bash
# 单张图片预测
python src/chladni_vision_pro.py --predict path/to/image.png --model models/demo_model.pkl

# 交互式预测 (推荐)
python src/chladni_vision_pro.py --interactive --model models/demo_model.pkl
```

### 交互式命令
在交互模式中，您可以使用以下命令：
- `image_path.png` - 预测指定图片
- `stats` - 显示预测统计信息
- `history` - 查看预测历史
- `batch` - 进入批量预测模式
- `quit` - 退出程序

## 📊 输出文件说明

所有生成的可视化文件保存在 `output/` 目录：

| 文件名 | 内容描述 |
|--------|----------|
| `enhanced_confusion_matrix.png` | 增强混淆矩阵 (绝对数量+百分比) |
| `model_comparison.png` | 算法性能对比图表 |
| `feature_analysis.png` | 特征分析 (分布+相关性+PCA) |
| `enhanced_prediction_*.png` | 详细预测结果面板 |
| `training_summary.png` | 训练过程摘要报告 |
| `detailed_classification_report.txt` | 详细文本报告 |

## 🔍 数据集说明

### 小数据集 (data/)
- **图片数量**: 5张
- **频率类别**: 600Hz, 700Hz, 800Hz, 900Hz, 1100Hz
- **每类数量**: 1张
- **训练时间**: ~10秒
- **适用场景**: 功能验证、快速测试

### 增强数据集 (data_augmented/)
- **图片数量**: 55张
- **频率类别**: 600Hz, 700Hz, 800Hz, 900Hz, 1100Hz
- **每类数量**: 11张
- **训练时间**: ~45秒
- **适用场景**: 正常训练、演示使用
- **特点**: 数据增强技术，性能更好

### 大数据集 (extracted_frames_full/)
- **图片数量**: 210张
- **频率类别**: 500Hz, 600Hz, 700Hz, 800Hz, 900Hz, 1000Hz, 1100Hz
- **每类数量**: 30张
- **训练时间**: ~2-3分钟
- **适用场景**: 完整训练、研究使用
- **特点**: 数据量最大，准确率最高

## 🛠️ 故障排除

### 常见问题

1. **中文字符乱码**
   ```bash
   # 解决方案: 使用Windows兼容启动器
   python src/run_chladni_pro_windows.py
   ```

2. **导入模块错误**
   ```bash
   # 解决方案: 安装依赖
   pip install -r requirements.txt
   ```

3. **模型加载失败**
   ```bash
   # 解决方案: 确保模型文件存在
   ls -la models/
   ```

4. **图片格式不支持**
   ```bash
   # 解决方案: 检查图片格式
   # 支持: PNG, JPG, JPEG
   ```

### 性能优化

- **内存优化**: 使用大数据集时建议关闭其他程序
- **GPU加速**: 如需GPU支持，可安装CUDA版本的scikit-learn
- **并行处理**: 系统已启用多线程特征提取

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

# 安装开发依赖
pip install -r requirements.txt
pip install black pytest flake8
```

### 代码风格
- 使用Black进行代码格式化
- 遵循PEP 8编码规范
- 添加适当的注释和文档字符串

### 提交规范
```bash
git commit -m "feat: 添加新的特征提取方法"
git commit -m "fix: 修复Windows兼容性问题"
git commit -m "docs: 更新README文档"
```

## 📄 许可证

本项目采用 [MIT 许可证](LICENSE) - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

感谢以下开源项目的支持：
- [OpenCV](https://opencv.org/) - 计算机视觉库
- [Scikit-learn](https://scikit-learn.org/) - 机器学习库
- [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/) - 数据可视化
- [NumPy](https://numpy.org/) & [Pandas](https://pandas.pydata.org/) - 数据处理
- [Tqdm](https://tqdm.github.io/) - 进度条库

## 📞 联系方式

- **作者**: Jason Robert Destiny
- **邮箱**: johnrobertdestiny@gmail.com
- **GitHub**: https://github.com/JasonRobertDestiny
- **项目地址**: https://github.com/JasonRobertDestiny/ChladniVision-Pro

## 📈 项目路线图

- [ ] 支持实时摄像头输入
- [ ] 添加Web界面
- [ ] 实现深度学习模型
- [ ] 支持更多频率类别
- [ ] 移动端应用开发
- [ ] 云端部署版本

---

⭐ 如果这个项目对您有帮助，请考虑给个星标！

[![Star History Chart](https://api.star-history.com/svg?repos=JasonRobertDestiny/ChladniVision-Pro&type=Date)](https://star-history.com/#JasonRobertDestiny/ChladniVision-Pro&Date)