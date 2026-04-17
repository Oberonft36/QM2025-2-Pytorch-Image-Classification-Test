# 试验报告

## 标准组实验配置：

#若图片无法打开，请见outputs文件夹

#### 实验文件：`train.py`

- train size: 60000
- test size: 10000
- class length: 10
- batch: 64
- images shape = [64, 1, 28, 28]
- labels shape: 64

- lr=0.001

#### 标准组最优准确度：

约为0.89

#### 输出：

```
Epoch [1/3] - train_loss: 0.5011, train_acc: 0.8180, test_acc: 0.8644
BEST model !!! = 0.8644
Epoch [2/3] - train_loss: 0.3190, train_acc: 0.8861, test_acc: 0.8831
BEST model !!! = 0.8831
Epoch [3/3] - train_loss: 0.2767, train_acc: 0.9000, test_acc: 0.8913
BEST model !!! = 0.8913
```

#### 标准组loss和acc曲线：

![loss_curve](https://github.com/Oberonft36/QM2025-2-Pytorch-Image-Classification-Test/blob/main/outputs/loss_curve.png?raw=true)

![acc_curve](https://github.com/Oberonft36/QM2025-2-Pytorch-Image-Classification-Test/blob/main/outputs/acc_curve.png?raw=true)

#若图片无法打开，请见outputs文件夹

## 对比实验1：学习率对比实验分析

#### 实验文件：`train_contrast_lr_change.py`

#### 实验配置：

- train size: 60000
- test size: 10000
- class length: 10
- batch: 64
- images shape = [64, 1, 28, 28]
- labels shape: 64

- lr=0.0001

#### 实验组最优准确度：

约为0.84

#### 本实验loss和acc曲线：

![loss_lr_change](https://github.com/Oberonft36/QM2025-2-Pytorch-Image-Classification-Test/blob/main/outputs/loss_lr_change.png?raw=true)

![acc_lr_change](https://github.com/Oberonft36/QM2025-2-Pytorch-Image-Classification-Test/blob/main/outputs/acc_lr_change.png?raw=true)

#若图片无法打开，请见outputs文件夹
#### 实验分析：

本实验对比了两种学习率设置（0.001 与 0.0001）对模型训练效果的影响。

从训练曲线可以看出，两种学习率下模型的 loss 均下降，同时accuracy 随 epoch 提升，说明模型均能够正常学习。然而学习率为 0.001 时，模型在较少的 epoch 内即可达到较高准确率；而学习率为 0.0001 时，最终准确率略低

在最终性能上，学习率 0.001 的模型测试准确率约为 0.89，而学习率 0.0001 的模型达到约 0.84，说明在当前训练轮数设置下，小学习率未能充分训练模型。

在本实验中，学习率 0.0001 导致模型更新步长较小，训练过程更加平稳。



## 对比实验2：batch对比实验分析

#### 实验文件：`train_contrast_batch_change.py`

#### 实验配置：

- train size: 60000
- test size: 10000
- class length: 10
- batch: 128
- images shape = [64, 1, 28, 28]
- labels shape: 64

- lr=0.001

#### 实验组最优准确度：

约为0.8696

#### 本实验loss和acc曲线：

![loss_batch_change](https://github.com/Oberonft36/QM2025-2-Pytorch-Image-Classification-Test/blob/main/outputs/loss_batch_change.png?raw=true)

![acc_batch_change](https://github.com/Oberonft36/QM2025-2-Pytorch-Image-Classification-Test/blob/main/outputs/acc_batch_change.png?raw=true)
)
#若图片无法打开，请见outputs文件夹
#### 实验分析：

本实验对比了不同 batch size（64 与 128）对模型训练效果的影响。

从训练 loss 曲线可以看出，batch size 为 128 时曲线更加平滑，说明较大的 batch 能够适度降低梯度，使训练过程更加稳定。然而在准确率曲线中可以观察到，当 batch size 为 128 时，测试集准确率在第 2 个 epoch 达到峰值后出现下降，而训练集准确率仍持续上升，表现：轻微过拟合现象。

相比之下，batch size 为 64 时，训练集与测试集准确率始终保持接近，且最终测试准确率更高，说明其泛化能力更好。

# 5 张错误分类样本及解释
#若图片无法打开，请见outputs文件夹
![error_samples](outputs\error_samples.png)
#若图片无法打开，请见outputs文件夹
## 错误样本分析

### 样本 1：`P:Sandal, T:Sneaker`

该样本真实类别是 Sneaker，但模型预测成了 Sandal。从图像上看，这只鞋的鞋帮较低，整体轮廓比较扁平，缺少高帮或厚底等明显特征。低分辨率导致细节信息很难保留下来，模型只根据“低矮、轻薄”判断，因此误判成了 Sandal。
这说明模型在鞋类分类时对局部结构特征提取不足。

### 样本 2：`P:Sandal, T:Ankle boot`

该样本真实类别是 Ankle boot，但被预测为 Sandal。这张图的成像比较模糊，鞋帮部分不够突出，整体也偏低矮。模型可能没有捕捉到“靴筒高度”这个关键点，从而做出错误判断。
这说明模型对鞋帮高度、鞋口形状等特征的学习还不够充分。

### 样本 3：`P:Pullover, T:Coat`

该样本真实类别是 Coat，预测为 Pullover。这两类整体轮廓相近。但在这张图中，细节并不清晰，模型更可能只学到了“长袖上衣”的特征，而没有学到 coat 的细节信息。
这表明模型虽然能识别“上衣”，但对上衣种类的区分能力不足。

### 样本 4：`P:Coat, T:Dress`

该样本真实类别是 Dress，预测为 Coat。从轮廓看，这件衣物整体呈现拉长的形状，上半部分较窄、下摆向下延摆，但由于图像清晰度有限，裙摆特征不够明显，被模型当作一件长外套。
这说明模型在服装识别中较依赖整体外轮廓，而对于 dress 和 coat 在下摆结构、衣身展开方式上的细节区别捕捉不够。

### 样本 5：`P:Trouser, T:Dress`

该样本真实类别是 Dress，却被预测成了 Trouser。这张图不像标准的 dress 轮廓，而是一个较窄的形态。模型很可能把中间的竖直区域当成裤腿结构，因此预测为 Trouser。
这说明当前模型对非典型样本的理解不足。



# 实验改进

### 1. 增强模型学习深度

增加卷积层数，或把通道增多一些，让模型提取更丰富的特征。

### 2. 增加训练轮数

现在的训练轮3轮数较少。增加 epoch 数，可以让模型进一步优化分类，尤其是相似类别之间的区分。

### 3. 尝试更强的数据处理方式

可以尝试适度的数据增强，如轻微平移、随机裁剪等，让模型减少对“标准轮廓模板”的依赖，提高对非常规样本的适应能力。
