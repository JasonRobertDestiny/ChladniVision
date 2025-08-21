# ChladniVision 🎵

一个专门用于克拉尼图形（Chladni Figures）分类的计算机视觉项目，使用Dense SIFT特征提取和KNN分类器实现不同频率声波产生的振动模式识别。

## 项目简介

ChladniVision是一个基于计算机视觉技术的克拉尼图形分类系统。克拉尼图形是由声波在平面上产生的美丽几何图案，不同频率的声波会产生不同的振动模式。本项目通过Dense SIFT特征提取和KNN分类算法，能够自动识别和分类不同频率下的克拉尼图形。

## 项目结构

```
ChladniVision/
├── data/                          # 克拉尼图形数据集
│   ├── 600Hz/                    # 600Hz频率图形
│   ├── 700Hz/                    # 700Hz频率图形
│   ├── 800Hz/                    # 800Hz频率图形
│   ├── 900Hz/                    # 900Hz频率图形
│   └── 1100Hz/                   # 1100Hz频率图形
├── utils/                         # 工具模块
│   ├── sift_extractor.py         # SIFT特征提取器
│   ├── knn_classifier.py         # KNN分类器
│   ├── data_preprocessing.py     # 数据预处理
│   └── evaluator.py              # 模型评估
├── demo.py                       # 主演示脚本
├── test_features.py              # 特征提取测试
├── requirements.txt              # 依赖包列表
├── 图片分类演示说明.md            # 中文说明文档
└── README.md                     # 项目说明
```

## 功能特性

### 🎵 克拉尼图形识别
- 支持多频率克拉尼图形分类（600Hz - 1100Hz）
- 基于物理声学原理的图案识别
- 适用于声学实验和教学演示

### 🔍 Dense SIFT特征提取
- **Dense SIFT**: 密集SIFT特征描述符，提供丰富的局部纹理信息
- **像素特征**: 传统像素级特征作为对比基准
- 自动特征标准化和归一化
- 2000维Dense SIFT vs 4096维像素特征

### 🤖 智能分类算法
- **KNN分类器**: K近邻算法，适合小样本学习
- **自适应数据分割**: 智能处理小数据集问题
- **特征标准化**: StandardScaler确保特征尺度一致
- **错误处理**: 完善的异常处理机制

### 📊 交互式演示
- 实时图像预测和置信度显示
- 支持中文界面和提示
- 图像可视化展示
- 特征提取对比分析

## 安装依赖

```bash
pip install -r requirements.txt
```

### 主要依赖包
- **OpenCV-Python**: 图像处理和SIFT特征提取
- **scikit-learn**: KNN分类器和数据预处理
- **matplotlib**: 图像显示和可视化
- **numpy**: 数值计算和数组操作
- **Pillow**: 图像文件读取和处理

## 快速开始

### 1. 运行克拉尼图形分类演示

```bash
# 运行主演示程序
python demo.py
```

演示程序将提供以下选项：
- 选择特征提取方法（Dense SIFT 或 像素特征）
- 自动训练KNN分类器
- 交互式图像预测

### 2. 测试特征提取效果

```bash
# 比较Dense SIFT和像素特征
python test_features.py
```

### 3. 数据集结构

项目使用的克拉尼图形数据集结构：
```
data/
├── 600Hz/
│   └── 600hz_001.png
├── 700Hz/
│   └── 700hz_001.png
├── 800Hz/
│   └── 800hz_001.png
├── 900Hz/
│   └── 900hz_001.png
└── 1100Hz/
    └── 1100hz_001.png
```

### 4. 预测单张图片

在演示程序中输入图片路径进行预测：
```
请输入图片路径: data/1100Hz/1100hz_001.png
```

## 详细使用说明

### Dense SIFT特征提取

```python
from demo import SimpleImageClassifier

# 初始化分类器（使用Dense SIFT）
classifier = SimpleImageClassifier(use_sift_features=True)

# 提取SIFT特征
features = classifier.extract_sift_features(image)
print(f"SIFT特征维度: {features.shape}")  # 输出: (2000,)
```

### KNN分类器训练

```python
# 加载数据并训练模型
classifier.load_images('data')
classifier.train_model()

# 预测新图像
prediction, confidence = classifier.predict_image('path/to/new_image.png')
print(f"预测结果: {prediction}, 置信度: {confidence:.2%}")
```

### 特征对比分析

```python
# 比较不同特征提取方法
from test_features import compare_features

image_path = 'data/1100Hz/1100hz_001.png'
sift_features, pixel_features = compare_features(image_path)

print(f"SIFT特征: 维度={sift_features.shape[0]}, 范围=[{sift_features.min()}, {sift_features.max()}]")
print(f"像素特征: 维度={pixel_features.shape[0]}, 范围=[{pixel_features.min()}, {pixel_features.max()}]")
```

## 技术特点

### Dense SIFT vs 传统像素特征

| 特征类型 | 维度 | 优势 | 适用场景 |
|----------|------|------|----------|
| **Dense SIFT** | 2000维 | 局部纹理丰富，旋转不变性 | 纹理复杂的克拉尼图形 |
| **像素特征** | 4096维 | 简单直接，计算快速 | 简单图案或对比基准 |

### 算法选择理由

- **KNN分类器**: 适合小样本学习，无需大量训练数据
- **Dense SIFT**: 提供丰富的局部纹理信息，适合克拉尼图形的复杂几何模式
- **特征标准化**: 确保不同尺度特征的公平比较

## 演示效果

### 分类结果示例

```
=== ChladniVision 克拉尼图形分类演示 ===
选择特征提取方法:
1. Dense SIFT 特征 (推荐)
2. 像素特征
请选择 (1-2): 1

正在加载图像数据...
找到 5 个类别: ['1100Hz', '600Hz', '700Hz', '800Hz', '900Hz']
加载了 5 张图像

使用 Dense SIFT 特征训练模型...
数据集较小(5个样本)，使用全部数据进行训练和测试
模型训练完成！准确率: 40.00%

预测结果: 1100Hz
置信度: 33.33%
```

### 特征提取对比

```
测试图像: data/1100Hz/1100hz_001.png
Dense SIFT 特征: 维度=2000, 范围=[0, 116], 均值=31.99
像素特征: 维度=4096, 范围=[0, 253], 均值=123.46
```

## 性能优化建议

### 1. 数据集扩充
- 收集更多不同频率的克拉尼图形样本
- 确保每个频率类别有足够的训练样本（建议≥20张）
- 包含不同实验条件下的图形变化

### 2. 特征优化
- 调整Dense SIFT参数（步长、描述符数量）
- 尝试其他局部特征描述符（ORB、SURF等）
- 结合全局特征和局部特征

### 3. 分类器调优
- 优化KNN的K值选择
- 尝试加权KNN或其他距离度量
- 考虑使用SVM或随机森林等其他分类器

### 4. 预处理改进
- 图像去噪和增强
- 标准化图像尺寸和对比度
- 背景去除和图形区域提取

## 常见问题

### Q: 为什么分类准确率较低？
A: 
- 当前数据集很小（每类只有1张图片），这是正常现象
- 建议收集更多样本来提高准确率
- Dense SIFT特征在小数据集上仍能提供有意义的分类信息

### Q: 如何添加新的频率类别？
A:
- 在 `data/` 目录下创建新的频率文件夹（如 `1200Hz/`）
- 添加对应频率的克拉尼图形图片
- 重新运行 `demo.py` 进行训练

### Q: SIFT特征提取失败怎么办？
A:
- 系统会自动回退到像素特征
- 检查OpenCV安装是否正确
- 确保图像文件格式正确且可读

### Q: 如何改进分类效果？
A:
- 增加每个类别的样本数量
- 尝试不同的特征提取参数
- 使用图像预处理技术
- 考虑集成多种特征

## 扩展功能

### 1. 添加新的特征提取器
在 `utils/` 目录下创建新的特征提取模块，如HOG、LBP等。

### 2. 集成深度学习模型
可以添加CNN模型作为特征提取器，与传统方法进行对比。

### 3. 实时图像分类
集成摄像头输入，实现实时克拉尼图形识别。

### 4. Web界面
开发基于Flask/Django的Web应用，提供在线分类服务。

## 参考资料

- [OpenCV SIFT文档](https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html)
- [scikit-learn KNN分类器](https://scikit-learn.org/stable/modules/neighbors.html#classification)
- [克拉尼图形物理原理](https://en.wikipedia.org/wiki/Chladni_figure)
- [Dense SIFT特征描述](https://www.vlfeat.org/overview/dsift.html)
- [计算机视觉特征提取方法](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_table_of_contents_feature2d/py_table_of_contents_feature2d.html)

## 项目背景

克拉尼图形是18世纪德国物理学家恩斯特·克拉尼（Ernst Chladni）发现的声学现象。当在撒有细沙的金属板上施加特定频率的声波时，沙粒会在振动节点聚集，形成美丽而复杂的几何图案。不同的频率会产生不同的图案，这为声学研究和艺术创作提供了独特的视角。

ChladniVision项目旨在通过计算机视觉技术自动识别和分类这些图案，为物理教学、声学研究和艺术创作提供技术支持。

## 许可证

本项目采用 MIT 许可证。

## 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目！特别欢迎：
- 新的克拉尼图形数据集
- 改进的特征提取算法
- 更好的分类器实现
- 用户界面优化

## 联系方式

如有问题或建议，请通过GitHub Issues联系我们。

---

**注意**: 这是一个教学和研究项目，用于演示计算机视觉在物理现象识别中的应用。项目展示了Dense SIFT特征提取和KNN分类在小样本学习中的效果。