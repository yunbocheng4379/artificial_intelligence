# 导入工具包
import numpy as np
import argparse
import imutils
import cv2

# 设置参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the input image")
args = vars(ap.parse_args())

# 正确答案
ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}


def order_points(pts):
    # 一共4个坐标点
    rect = np.zeros((4, 2), dtype="float32")

    # 按顺序找到对应坐标0123分别是 左上，右上，右下，左下
    # 计算左上，右下
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 计算右上和左下
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    # 获取输入坐标点
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 计算输入的w和h值
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 变换后对应坐标位置
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # 计算变换矩阵
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # 返回变换后结果
    return warped


"""
    该方法用于【对轮廓进行排序】，支持四种排序方式。
    输入：轮廓列表cnts和排序方式method
    输出：排序后的轮廓和对应的外接矩形框
    排序方式：
        left-to-right（从左到右，默认）
        right-to-left（从右到左）
        top-to-bottom（从上到下）
        bottom-to-top（从下到上）
"""


def sort_contours(cnts, method="left-to-right"):
    # 升序
    reverse = False
    # x轴
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        # 降序
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        # y轴
        i = 1
    # 根据传递的参数判断是按照x轴排序还是按照y轴排序、升序还是降序
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
    return cnts, boundingBoxes


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 原始图像预处理
# 读取原始图像
image = cv2.imread(args["image"])
# 复制原始图像，防止破坏原始图像
contours_img = image.copy()
# BGR -> 灰度
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 高斯模糊（平滑处理）
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# 展示图像
cv_show('blurred', blurred)
# 边缘检测
edged = cv2.Canny(blurred, 75, 200)
# 展示图像
cv_show('edged', edged)

# 轮廓检测
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
# -1：，表示将全部轮廓进行绘制
# (0, 0, 255)：是BGR格式，转换为RGB就是(255, 0, 0)代表红色
# 3：代表绘制轮廓线段的粗细
cv2.drawContours(contours_img, cnts, -1, (0, 0, 255), 3)
cv_show('contours_img', contours_img)
docCnt = None

# 确保检测到了
if len(cnts) > 0:
    # 按轮廓面积从大到小排序（cv2.contourArea计算面积，reverse=True降序）
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    # 遍历轮廓并近似多边形
    for c in cnts:
        # cv2.arcLength(c, True)：计算轮廓周长（True表示闭合轮廓）
        peri = cv2.arcLength(c, True)
        # cv2.approxPolyDP(c, epsilon, True)：多边形近似
        # epsilon = 0.02 * peri：近似精度（周长2%的容差）
        # True：表示近似曲线是闭合的
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # 准备做透视变换
        # 如果近似多边形有4个顶点（len(approx)==4），则判定为四边形
        # 保存到docCnt并终止遍历（因已按面积排序，第一个找到的四边形是最大候选）
        # 在本项目中最大的四边形就是整个答题卡
        if len(approx) == 4:
            docCnt = approx
            break

# 执行透视变换
# 透视变换：将图像中倾斜或扭曲区域（如文档、车牌等）矫正为正面视图。
warped = four_point_transform(gray, docCnt.reshape(4, 2))
# 展示图像
cv_show('warped', warped)
# Otsu's 阈值处理（生成二值图）
# cv2.THRESH_OTSU：自动计算图像的最佳分割阈值，适用于灰度直方图呈双峰分布的图像。
thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# 展示图像
cv_show('thresh', thresh)
# 复制图像
thresh_Contours = thresh.copy()
# 找到每一个圆圈轮廓
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
# 绘制轮廓
cv2.drawContours(thresh_Contours, cnts, -1, (0, 0, 255), 3)
# 展示图像
cv_show('thresh_Contours', thresh_Contours)

questionCnts = []
# 遍历
for c in cnts:
    # 计算比例和大小
    # 获取边界矩形，外接矩形
    # x,y：表示该轮廓最小外接矩形左上角顶点在图像坐标系中的坐标。
    # w(宽度)和y(高度)：矩形的宽度（横向跨度）和高度（纵向跨度），从(x,y)开始向右和向下延伸。
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)

    # 根据实际情况指定标准
    # 判断哪些轮廓为答题卡选择题填图答案部分
    if w >= 20 and h >= 20 and 0.9 <= ar <= 1.1:
        questionCnts.append(c)

# 按照从上到下进行排序
questionCnts = sort_contours(questionCnts, method="top-to-bottom")[0]

# 存储回答正确的数量
correct = 0
# 每排有5个选项
for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
    # 按照从左到右排序（默认）
    cnts = sort_contours(questionCnts[i:i + 5])[0]
    bubbled = None

    # 遍历每一个结果
    for (j, c) in enumerate(cnts):
        # 通过轮廓掩膜（mask）技术，从多个候选区域中筛选出填充最完整的选项（例如答题卡涂卡识别）
        # 创建纯黑掩膜：生成一个与二值图 thresh 同尺寸的全黑图像（所有像素值为0），作为掩膜画布。
        mask = np.zeros(thresh.shape, dtype="uint8")
        # 绘制填充轮廓
        # [c]：cv2.drawContours要求传递的轮廓必须是一个列表（哪怕只有一个轮廓，也需要封装成列表）
        # 255：填充颜色（白色）
        # 第一个-1：将[c]中全部轮廓进行绘制
        # 第二个-1：厚度设为-1表示填充轮廓内部
        # 效果：在纯黑掩膜上，将当前轮廓区域填充为白色，其余部分仍为黑色。
        cv2.drawContours(mask, [c], -1, 255, -1)
        # 可视化掩膜，检查轮廓填充是否正确（白色区域=候选答案区域)
        cv_show('mask', mask)
        # 提取目标区域像素：将二值图 thresh 与掩膜 mask 结合（与操作，一零则零），仅保留掩膜白色区域对应的原图像素。
        # 生成的新图像中，只有当前轮廓内部的原图像素被保留，其余区域变为黑色。
        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        cv_show('mask', mask)
        # 计算掩膜区域内非零像素（即涂卡标记）的数量，数值越大表示填充越完整。
        total = cv2.countNonZero(mask)

        # 通过阈值判断，存储每行中填图的答案
        if bubbled is None or total > bubbled[0]:
            bubbled = (total, j)

    # 对比正确答案
    # 红色
    color = (0, 0, 255)
    # q是索引值，从0开始
    k = ANSWER_KEY[q]

    # 判断正确
    if k == bubbled[1]:
        # 绿色
        color = (0, 255, 0)
        correct += 1

    # 绘制回答正确的轮廓（正确绘制绿色轮廓，错误绘制红色轮廓）
    cv2.drawContours(warped, [cnts[k]], -1, color, 3)

# 计算回答准确率
score = (correct / 5.0) * 100
print("[INFO] score: {:.2f}%".format(score))
# 将准确率标注在图像上
cv2.putText(warped, "{:.2f}%".format(score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
# 展示原始图像
cv2.imshow("Original", image)
# 展示标注正确的图像
cv2.imshow("Exam", warped)
cv2.waitKey(0)
