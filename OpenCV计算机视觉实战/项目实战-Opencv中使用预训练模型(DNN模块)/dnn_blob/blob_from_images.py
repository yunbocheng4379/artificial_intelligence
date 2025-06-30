"""
OpenCV中的dnn模块提供了一套完整的工具链，用于加载预训练模型、预处理输入数据、并执行推理/预测。

OpenCV DNN模块核心功能
- 支持多种框架的模型：通过cv2.dnn.readNetFrom...()系列函数直接加载 Caffe/TensorFlow/PyTorch/ONNX等框架的预训练模型。
	net = cv2.dnn.readNetFromCaffe("model.prototxt", "model.caffemodel")  # Caffe
	net = cv2.dnn.readNetFromONNX("model.onnx")                           # ONNX

- 数据预处理：使用 cv2.dnn.blobFromImage()【单张】 或 blobFromImages()【批量】 将输入图像转换为模型所需的 标准化Blob格式（包括缩放、归一化、通道顺序调整等）。

- 高效推理：通过 net.setInput(blob) 和 net.forward() 执行前向传播，获取预测结果。


"""

# 导入工具包
import utils_paths
import numpy as np
import cv2

# 标签文件处理
rows = open("synset_words.txt").read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

# OpenCV的dnn模块加载一个Caffe预训练深度学习模型，用于后续的推理（预测）。
# cv2.dnn.readNetFromCaffe()：用于加载Caffe框架训练的模型
# 参数：
# "bvlc_googlenet.prototxt" → 模型结构配置文件（定义网络架构，如层类型、输入尺寸等）。
# "bvlc_googlenet.caffemodel" → 训练好的模型权重文件（包含训练后的参数）。
# 返回值：
# cv2.dnn_Net 对象，可以用于后续的推理（如目标检测、图像分类等）。

# bvlc_googlenet是GoogleNet的一个变种，是一个经典的深度卷积神经网络（CNN），常用于图像分类任务。
net = cv2.dnn.readNetFromCaffe("bvlc_googlenet.prototxt", "bvlc_googlenet.caffemodel")

# 获取图像路径
# 对列表[imagePaths]中的图片路径按 字母顺序排序（确保每次运行的顺序一致）
imagePaths = sorted(list(utils_paths.list_images("images/")))

# 图像数据预处理
# 读取图像
image = cv2.imread(imagePaths[0])
# 重置图像大小
resized = cv2.resize(image, (224, 224))
# 使用OpenCV的dnn模块对图像进行预处理，生成一个适合输入到深度学习模型的blob（Binary Large Object，二进制层对象）
# 参数说明：
# resized：输入图像（通常已经缩放到固定尺寸，如(244, 244)，要求格式为：H x W X c(高度 x 宽度 x 通道，OpenCV默认是BGR顺序)）
# 1[scalefactor]：像素值缩放因子，通常用于归一化（例如1/255.0像素值从[0, 255]缩放到[0, 1]）
# (244, 244)[size]：模型期望的输入尺寸（需与模型定义一致，如GoogleNet、ResNet 等经典 CNN 通常使用 224x224）
# (104, 117, 123)[mean]：均值减法，用于标准化。通常是BGR三通道的均值，计算方式为：image - mean，如果不需要均值减法，可设为(0, 0, 0)
# 重点：以上这些参数必须和引用的预训练模型在训练时指定的参数一致。比如：本例子中使用的是Caffe预训练模型，其中在训练时指定均值减法为(104, 117, 123)
# 所以在验证时也必须指定均值减法为同样的值

# 输出说明：生成的blob是一个4D NumPy数组，形状为(1, 3, 244, 244)
# 1：Batch size（批量大小，单张图像时为1）
# 3：通道数（BGR顺序）
# 244, 244：图像高度和宽度

# blobFromImage()函数用于处理单个图像
blob = cv2.dnn.blobFromImage(resized, 1, (224, 224), (104, 117, 123))
print("First Blob: {}".format(blob.shape))

# 将预处理后的图像输入到神经网络模型中
net.setInput(blob)
# 执行前向转播（推理）得到预测结果
preds = net.forward()

# 获取预测概率最高的类别索引
idx = np.argsort(preds[0])[::-1][0]
# 生成标注文本
text = "Label: {}, {:.2f}%".format(classes[idx], preds[0][idx] * 100)
# 将文本绘制到图像上
cv2.putText(image, text, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# 显示图像
cv2.imshow("Image", image)
cv2.waitKey(0)

# Batch图像数据制作
images = []
# 方法一样，数据是一个batch
for p in imagePaths[1:]:
	image = cv2.imread(p)
	image = cv2.resize(image, (224, 224))
	images.append(image)

# blobFromImages()函数用于批量处理图像，注意与处理单张图像多出来一个s
blob = cv2.dnn.blobFromImages(images, 1, (224, 224), (104, 117, 123))
print("Second Blob: {}".format(blob.shape))

# 获取预测结果
net.setInput(blob)
preds = net.forward()
for (i, p) in enumerate(imagePaths[1:]):
	image = cv2.imread(p)
	idx = np.argsort(preds[i])[::-1][0]
	text = "Label: {}, {:.2f}%".format(classes[idx], preds[i][idx] * 100)
	cv2.putText(image, text, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	cv2.imshow("Image", image)
	cv2.waitKey(0)