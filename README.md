# dl-2026-02 图像分类项目

## 1. 项目简介

本项目基于 PyTorch 实现一个基础图像分类任务，完成从数据加载、模型训练、模型评估到误差分析的完整流程。

目标：

- 实现完整训练流程
- 能复现实验结果
- 能分析模型错误并提出改进方向



## 2. 数据集说明

使用 **Fashion-MNIST**

- 输入尺寸：`1 × 28 × 28`
- 类别数：`10`
- 训练集：60000
- 测试集：10000
- batch size：`64`

类别：

```
T-shirt/top, Trouser, Pullover, Dress, Coat,
Sandal, Shirt, Sneaker, Bag, Ankle boot
```



## 3. 项目结构

```
.
│  error_test.py
│  evaluate.py
│  experiment_report.md
│  models.py
│  README.md
│  train.py
│  train_contrast_batch_change.py
│  train_contrast_lr_change.py
│
├─.idea
│  │  .gitignore
│  │  lpytorch.iml
│  │  misc.xml
│  │  modules.xml
│  │  workspace.xml
│  │
│  └─inspectionProfiles
│          profiles_settings.xml
│
├─checkpoints
│      best.pt
│      best_batch_change.pt
│      best_lr_change.pt
│
├─data
│  └─FashionMNIST
│      └─raw
│              t10k-images-idx3-ubyte
│              t10k-images-idx3-ubyte.gz
│              t10k-labels-idx1-ubyte
│              t10k-labels-idx1-ubyte.gz
│              train-images-idx3-ubyte
│              train-images-idx3-ubyte.gz
│              train-labels-idx1-ubyte
│              train-labels-idx1-ubyte.gz
│
├─outputs
│      acc_batch_change.png
│      acc_curve.png
│      acc_lr_change.png
│      error_samples.png
│      loss_batch_change.png
│      loss_curve.png
│      loss_lr_change.png
│      samples.png
│
├─POW images
│      获取数据集信息.png
│      训练证明.png
│
└─__pycache__
        models.cpython-310.pyc
```



## 4. 数据加载与预处理

- 使用 `torchvision.datasets` 加载数据
- 使用 `DataLoader` 构建批处理
- transform：`ToTensor()`



## 5. 模型设计

实现一个基础分类模型(CNN)：

- 输入: 灰度图
- 输出：10 类
- 支持 GPU（cuda）



## 6. 训练配置

- 损失函数：CrossEntropyLoss
- 优化器：Adam
- 学习率：0.001 
- epoch：3
- device：自动选择 cpu / cuda



## 7. 训练结果

常规组

```
Epoch [1/3] - train_loss: 0.5011, train_acc: 0.8180, test_acc: 0.8644
BEST model !!! = 0.8644
Epoch [2/3] - train_loss: 0.3190, train_acc: 0.8861, test_acc: 0.8831
BEST model !!! = 0.8831
Epoch [3/3] - train_loss: 0.2767, train_acc: 0.9000, test_acc: 0.8913
BEST model !!! = 0.8913
```



## 8. 模型保存

- 保存策略：test_acc 提升时保存
- 路径：

```
checkpoints/best.pt
```



## 9. 模型评估

运行：

```
python evaluate.py
```

功能：

- 加载 best.pt
- 输出测试集准确率



## 10. 对比实验

### 学习率对比

| 学习率 | test_acc |
| ------ | -------- |
| 0.001  | 0.89     |
| 0.0001 | 0.84     |

### Batch对比

| Batch | test_acc |
| ----- | -------- |
| 64    | 0.89     |
| 128   | 0.87     |



## 11. 训练曲线

路径：

```
outputs/loss_curve.png
```

作用：

- 观察收敛情况
- 判断是否过拟合



## 12. 错误分析

选取了 5 个错误样本进行分析：

分析内容包括：

- 真实标签和预测标签
- 错误原因



## 13. 改进方向

- 数据增强
- 更深的卷积模型
- 调整学习率



## 14. 复现实验

### 安装依赖

```
pip install torch torchvision matplotlib
```

### 训练

```
python train.py
```

### 评估

```
python evaluate.py
```



## 15. 总结

本项目完成了：

- 数据加载与可视化
- 模型训练与验证
- checkpoint 保存与加载
- 对比实验
- 训练曲线分析
- 错误样本分析

