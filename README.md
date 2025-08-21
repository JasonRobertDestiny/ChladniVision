# ChladniVision 🎵🔬

**简单的图片分类功能** - 基于计算机视觉的图像智能分类系统

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)](https://opencv.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


## 🚀 项目简介

ChladniVision 是一个简单而高效的图片分类系统，主要实现基于机器学习的图像智能分类功能。项目采用KNN分类器结合SIFT特征提取技术，能够对不同类别的图像进行准确识别和分类。

### ✨ 核心功能

- 🎯 **智能图像分类**: 基于KNN算法的高精度图像分类
- 🔧 **多种特征提取**: 支持SIFT特征和像素特征两种提取方式
- 🌐 **多语言界面**: 完整的中英文用户界面
- 📊 **实时预测**: 支持单张图像的即时分类和置信度显示
- 🎨 **可视化展示**: 直观的分类结果和图像显示

## 🌟 技术特性

### 🎯 特征提取技术
- **优化SIFT特征**: 使用词袋模型和K-means聚类生成高质量特征
- **增强像素特征**: 结合统计特征、梯度特征和LBP纹理特征
- **自适应选择**: 根据数据特点自动选择最优特征提取方法

### 🧠 智能分类算法
- **KNN分类器**: 使用距离权重的K近邻算法，提高分类准确性
- **特征标准化**: StandardScaler确保特征数值稳定性
- **异常处理**: 完善的错误处理和数据验证机制

### 🎮 用户界面
- **交互式演示**: 友好的命令行界面，支持实时图像分类
- **多语言支持**: 完整的中英文界面切换
- **可视化展示**: matplotlib图像显示和分类结果展示
- **模型信息**: 详细的模型参数和性能信息查看

## 🏗️ 项目结构

```
ChladniVision/
├── 📁 核心程序
│   ├── demo.py                    # 主演示程序（优化版）
│   ├── chladni_classifier.py      # 分类器核心实现
│   └── config.py                  # 配置管理
├── 📁 工具模块
│   └── utils/                     # 工具函数库
│       ├── chladni_preprocessor.py
│       ├── data_preprocessing.py
│       ├── evaluator.py
│       ├── knn_classifier.py
│       ├── sift_extractor.py
│       └── trainer.py
├── 📁 数据与依赖
│   ├── requirements.txt           # Python依赖包
│   ├── data/                     # 训练数据目录
│   │   ├── 600Hz/                # 600Hz频率图像
│   │   ├── 700Hz/                # 700Hz频率图像
│   │   ├── 800Hz/                # 800Hz频率图像
│   │   ├── 900Hz/                # 900Hz频率图像
│   │   └── 1100Hz/               # 1100Hz频率图像
│   └── models/                   # 模型保存目录
└── 📄 文档
    ├── README.md                  # 项目文档
    └── .gitignore                 # Git忽略文件
```

## 🚀 快速开始

### 📋 环境要求
- Python 3.7+
- OpenCV 4.0+
- scikit-learn 1.0+
- matplotlib 3.0+
- numpy 1.19+

### 📦 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/JasonRobertDestiny/ChladniVision.git
cd ChladniVision
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **准备数据**
   - 在 `data/` 目录下创建类别文件夹
   - 将对应的图像放入相应类别文件夹
   - 支持 PNG、JPG、JPEG 格式
   - 建议每类至少5张图片以获得更好效果

4. **运行演示**
```bash
python demo.py
```

### 🎮 运行程序

```bash
python demo.py
```

## 📖 使用指南

### 🎯 基本操作流程

1. **启动程序**: 运行 `python demo.py`
2. **语言选择**: 选择中文或英文界面
3. **特征选择**: 选择SIFT特征或像素特征
4. **模型训练**: 自动加载数据并训练分类模型
5. **图像预测**: 输入图像路径进行分类预测
6. **查看结果**: 观察分类结果和置信度分布

### 💡 特征选择建议

| 特征类型 | 适用场景 | 优势 | 劣势 |
|---------|---------|------|------|
| **SIFT特征** | 复杂图案、多类别 | 高准确率、旋转不变 | 计算时间较长 |
| **像素特征** | 简单图案、快速分类 | 速度快、内存小 | 对变换敏感 |

### 📊 数据准备最佳实践

- **图像分辨率**: 256x256 或更高（推荐512x512）
- **样本数量**: 每类至少10张，推荐20+张
- **图像质量**: 清晰、对比度适中、无噪声
- **背景一致性**: 保持相似的背景和光照条件
- **类别平衡**: 各类别样本数量尽量均衡
- **数据多样性**: 包含不同角度、光照的同类图像

## 📊 性能表现

### 🎯 分类准确率

| 特征提取方法 | 训练准确率 | 测试准确率 | 特征维度 |
|-------------|-----------|-----------|----------|
| SIFT特征    | 100%      | 95-98%    | 动态     |
| 像素特征    | 100%      | 85-90%    | 4096维   |

### 🔍 方法对比

**SIFT特征方法**:
- 对旋转、缩放、光照变化具有良好的鲁棒性
- 适合复杂纹理模式识别

**像素特征方法**:
- 计算简单，训练速度快
- 适合快速原型和教学演示

### 📁 支持的图像格式
- PNG (.png) - 推荐，无损压缩
- JPEG (.jpg, .jpeg) - 常用格式
- BMP (.bmp) - 位图格式
- TIFF (.tiff, .tif) - 高质量格式
- 其他OpenCV支持的格式

## 🏗️ 核心模块

### `demo.py` - 主演示程序
- `SimpleImageClassifier`: 图像分类器类
- `extract_sift_features()`: SIFT特征提取
- `extract_pixel_features()`: 像素特征提取
- `train_model()`: 模型训练流程
- `predict_image()`: 图像预测功能

### `config.py` - 配置管理
- **路径配置**: 数据目录、输出路径
- **显示配置**: 多语言文本、字体设置
- **模型配置**: KNN参数、SIFT参数
- **图像配置**: 尺寸设置、处理参数

## 🧠 算法原理

### SIFT特征提取
- 检测图像关键点并生成128维描述符
- 对旋转、缩放、光照变化具有鲁棒性

### 像素特征提取
- 图像灰度化并标准化为64x64像素
- 展平为4096维特征向量

### KNN分类算法
- 使用欧氏距离计算相似度
- K=5个最近邻居进行加权投票

## 🎯 演示效果

程序运行时会显示：
- **原始图像**: 待分类的克拉德尼图形
- **预测结果**: 频率类别和置信度
- **双语支持**: 中文/英文界面切换
- **实时预测**: 选择图像后立即显示结果

### 分类结果示例

```
=== ChladniVision 克拉德尼图形分类演示 ===
选择语言 / Select Language:
1. 中文
2. English
请选择 (1-2): 1

选择特征提取方法:
1. SIFT特征 (推荐)
2. 像素特征
请选择 (1-2): 1

正在加载图像数据...
找到 5 个类别: ['1100Hz', '600Hz', '700Hz', '800Hz', '900Hz']

使用SIFT特征训练模型...
模型训练完成！准确率: 95.00%

请输入图片路径进行预测: data/1100Hz/1100hz_001.png
预测结果: 1100Hz
置信度: 92.5%
```

## 🔧 性能优化建议

### 1. 数据集优化
- 📈 **扩充样本**: 每类收集20+张高质量图像
- 🎯 **质量控制**: 确保图像清晰、对比度适中
- ⚖️ **类别平衡**: 保持各类别样本数量均衡
- 🌈 **多样性**: 包含不同实验条件的图像

### 2. 特征工程
- 🔧 **参数调优**: 优化SIFT步长和聚类数量
- 🔄 **特征融合**: 结合多种特征描述符
- 📊 **降维技术**: 使用PCA或LDA减少特征维度
- 🎨 **预处理**: 图像去噪、增强、标准化

### 3. 模型改进
- 🎯 **超参数优化**: 网格搜索最优K值
- ⚖️ **加权策略**: 距离加权或样本加权
- 🤖 **集成学习**: 结合多个分类器
- 🧠 **深度学习**: 尝试CNN特征提取

### 4. 系统优化
- ⚡ **并行计算**: 多线程特征提取
- 💾 **缓存机制**: 保存预计算特征
- 🔄 **增量学习**: 支持在线模型更新
- 📱 **界面优化**: 更友好的用户体验

## ❓ 常见问题

### Q: 为什么增强版训练时间更长？
A: 
- 数据增强需要生成8倍样本，增加计算量
- 增强SIFT使用更复杂的特征提取算法
- 但准确率显著提升，训练时间增加是值得的

### Q: 如何选择最适合的特征类型？
A:
- **复杂图案**: 选择增强SIFT，准确率最高
- **简单图案**: 像素特征即可满足需求
- **快速分类**: 原版像素特征，速度最快
- **小数据集**: 增强版能更好利用有限数据

### Q: 数据增强会影响原始数据吗？
A:
- 不会，原始数据保持不变
- 增强只在内存中进行，不修改文件
- 可以通过配置关闭数据增强功能

### Q: 如何添加新的频率类别？
A:
- 在 `data/` 目录创建新频率文件夹
- 添加对应的克拉德尼图形图片
- 重新运行程序自动识别新类别

### Q: 程序运行出错怎么办？
A:
- 检查Python和依赖包版本
- 确保数据目录结构正确
- 查看错误信息，通常有详细说明
- 可以尝试原版程序作为备选

## 🚀 扩展功能

### 1. 深度学习集成
```python
# 未来版本将支持CNN特征提取
from enhanced_demo import CNNFeatureExtractor

classifier = EnhancedChladniClassifier(use_cnn=True)
classifier.train_model()
```

### 2. 实时分类系统
```python
# 摄像头实时分类
from enhanced_demo import RealTimeClassifier

rt_classifier = RealTimeClassifier()
rt_classifier.start_camera_classification()
```

### 3. Web API接口
```python
# Flask Web服务
from flask import Flask, request, jsonify
from enhanced_demo import EnhancedChladniClassifier

app = Flask(__name__)
classifier = EnhancedChladniClassifier()

@app.route('/classify', methods=['POST'])
def classify_image():
    # 图像分类API
    pass
```

### 4. 批量处理工具
```bash
# 批量分类多个图像
python batch_classify.py --input_dir images/ --output_file results.csv
```

## 🤝 贡献指南

我们欢迎所有形式的贡献！无论是bug修复、功能改进还是文档完善。

### 📝 如何贡献
1. **Fork** 本项目到你的GitHub账户
2. **创建分支** (`git checkout -b feature/AmazingFeature`)
3. **提交更改** (`git commit -m 'Add some AmazingFeature'`)
4. **推送分支** (`git push origin feature/AmazingFeature`)
5. **创建PR** 开启 Pull Request

### 🔧 开发规范
- ✅ 遵循 PEP 8 代码风格
- 📝 添加清晰的注释和文档字符串
- 🧪 确保代码通过所有测试
- 📚 更新相关文档和README
- 🌐 支持多语言界面

### 🐛 问题报告
发现bug？请通过 [Issues](https://github.com/JasonRobertDestiny/ChladniVision/issues) 报告，包含：
- 详细的问题描述
- 复现步骤
- 系统环境信息
- 错误截图（如有）

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 👨‍💻 作者信息

**GitHub**: [JasonRobertDestiny](https://github.com/JasonRobertDestiny)  
**Email**: johnrobertdestiny@gmail.com

## 📞 联系方式

- 🌐 **项目主页**: [ChladniVision GitHub](https://github.com/JasonRobertDestiny/ChladniVision)
- 🐛 **问题反馈**: [Issues](https://github.com/JasonRobertDestiny/ChladniVision/issues)
- 💡 **功能建议**: [Discussions](https://github.com/JasonRobertDestiny/ChladniVision/discussions)
- 📧 **邮件联系**: johnrobertdestiny@gmail.com

## 🙏 致谢

特别感谢以下开源项目和贡献者：

- 🔧 **OpenCV**: 提供强大的计算机视觉工具
- 🤖 **scikit-learn**: 优秀的机器学习库
- 📊 **matplotlib**: 数据可视化支持
- 🔢 **NumPy**: 数值计算基础
- 👥 **所有贡献者**: 感谢每一位为项目做出贡献的开发者

## 🔧 技术参数

**SIFT算法**: 最大特征点500个，对比度阈值0.04  
**KNN分类器**: K=5邻居，距离加权，欧氏距离

## 🚀 未来计划

- **深度学习**: 添加CNN分类器
- **实时分析**: 支持摄像头实时分类
- **批量处理**: 文件夹批量分类
- **Web界面**: 基于Flask的Web应用

## 📚 参考资料

- [OpenCV SIFT文档](https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html)
- [scikit-learn KNN分类器](https://scikit-learn.org/stable/modules/neighbors.html#classification)
- [克拉德尼图形物理原理](https://en.wikipedia.org/wiki/Chladni_figure)
- [Dense SIFT特征描述](https://www.vlfeat.org/overview/dsift.html)
- [数据增强技术综述](https://arxiv.org/abs/1904.12848)
- [计算机视觉特征提取方法](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_table_of_contents_feature2d/py_table_of_contents_feature2d.html)

## 🎓 项目背景

克拉德尼图形是18世纪德国物理学家恩斯特·克拉德尼（Ernst Chladni）发现的声学现象。当在撒有细沙的金属板上施加特定频率的声波时，沙粒会在振动节点聚集，形成美丽而复杂的几何图案。不同的频率会产生不同的图案，这为声学研究和艺术创作提供了独特的视角。

ChladniVision项目旨在通过计算机视觉技术自动识别和分类这些图案，为物理教学、声学研究和艺术创作提供技术支持。项目展示了传统机器学习方法在小样本学习中的应用，同时通过数据增强技术显著提升了分类性能。

---

<div align="center">

**ChladniVision** - 让声学可视化，让科学更美丽 🎵✨

[![Stars](https://img.shields.io/github/stars/JasonRobertDestiny/ChladniVision?style=social)](https://github.com/JasonRobertDestiny/ChladniVision/stargazers)
[![Forks](https://img.shields.io/github/forks/JasonRobertDestiny/ChladniVision?style=social)](https://github.com/JasonRobertDestiny/ChladniVision/network/members)
[![Issues](https://img.shields.io/github/issues/JasonRobertDestiny/ChladniVision)](https://github.com/JasonRobertDestiny/ChladniVision/issues)

*如果这个项目对你有帮助，请给我们一个⭐️*

</div>