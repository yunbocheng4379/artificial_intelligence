# 处理图像中的轮廓，包含对轮廓进行排序、筛选、坐标调整等
from imutils import contours
import numpy as np
import argparse
import cv2
import myutils

# 设置参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
ap.add_argument("-t", "--template", required=True,
                help="path to template OCR-A image")
args = vars(ap.parse_args())

# 指定信用卡类型
FIRST_NUMBER = {
    "3": "American Express",
    "4": "Visa",
    "5": "MasterCard",
    "6": "Discover Card"
}


# 绘图展示
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 读取一个模板图像
img = cv2.imread(args["template"])
cv_show('img', img)
# 灰度图（BGR[三通道] -> gray[单通道]）
ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv_show('ref', ref)
# 二值图像（cv2.THRESH_BINARY_INV：超过阈值[10]部分取0，否则取最大值[255]）
# 之索引这样转换是因为给 cv2.findContours() 显式提供二值图（边缘为白色，背景为黑色的图像）
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]
cv_show('ref', ref)

# 计算轮廓
# cv2.findContours()函数接受的参数为二值图，即黑白的（不是灰度图）,cv2.RETR_EXTERNAL只检测外轮廓，cv2.CHAIN_APPROX_SIMPLE只保留终点坐标
# 返回的list中每个元素都是图像中的一个轮廓

# 这个地方使用 ref.copy() ，保证原图不变
refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# -1参数，表示将全部轮廓进行绘制
cv2.drawContours(img, refCnts, -1, (0, 0, 255), 3)
cv_show('img', img)

# 在进行轮廓绘制时不需要关注轮廓的顺序，只是将集合中全部的轮廓按照位置进行全部绘制即可
# 最终生成的轮廓并一定是按照 0~9 的顺序的，进行排序，从左到右，从上到下
# 按照每个轮廓左上角的坐标进行排序，从小到大排序一下就是最终从0~9顺序的轮廓
refCnts = myutils.sort_contours(refCnts, method="left-to-right")[0]
digits = {}

# 遍历每一个轮廓
# c是每一个轮廓，i是每个轮廓的索引
for (i, c) in enumerate(refCnts):
    # 获取轮廓边界位置信息
    (x, y, w, h) = cv2.boundingRect(c)
    # 绘制矩形边框，制作模板
    roi = ref[y:y + h, x:x + w]
    # 重置模板尺寸，设置为 (57, 88)
    # 必须将模板(roi)和待检测目标的尺寸统一调整为固定值（如57X88），这是模板匹配能够正确工作的核心前提。
    roi = cv2.resize(roi, (57, 88))
    # 每一个索引对应每一个模板
    # 0 -> 检测0数字模板，1 -> 检测1数据模板...
    digits[i] = roi

# 初始化卷积核
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# 读取输入图像，预处理
image = cv2.imread(args["image"])
cv_show('image', image)
# 修改输入图像大小
image = myutils.resize(image, width=300)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv_show('gray', gray)

# 礼帽操作，突出更明亮的区域
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
cv_show('tophat', tophat)

# 边缘检测：dx=1,dy=0代表目前计算的为水平的（右侧减左侧），不计算竖直的
# ksize=-1 相当于用3*3的
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
# 获取绝对值
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")
print(np.array(gradX).shape)
cv_show('gradX', gradX)

# 通过闭操作（先膨胀，再腐蚀）将数字连在一起
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
cv_show('gradX', gradX)
# 二值图
# THRESH_OTSU会自动寻找合适的阈值，适合双峰，需把阈值参数设置为0
thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv_show('thresh', thresh)

# 通过闭操作（先膨胀，在腐蚀）将数字中的空白区域再次去除
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
cv_show('thresh', thresh)

# 计算轮廓
threshCnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 存储计算后的轮廓信息
cnts = threshCnts
# 复制原始图像，防止原始图像被破坏
cur_img = image.copy()
# 传入绘制图像，轮廓，轮廓索引，颜色模式，线条厚度
cv2.drawContours(cur_img, cnts, -1, (0, 0, 255), 3)
cv_show('img', cur_img)

# 遍历轮廓，排除不符合条件的轮廓数据
# 通过宽高比来过滤掉不符合条件的轮廓
locs = []
for (i, c) in enumerate(cnts):
    # 计算矩形高和宽
    (x, y, w, h) = cv2.boundingRect(c)
    # 计算宽高比
    ar = w / float(h)

    # 选择合适的区域，根据实际任务来，目前是四个数字一组
    if 2.5 < ar < 4.0:
        if (40 < w < 55) and (10 < h < 20):
            # 符合的留下来
            locs.append((x, y, w, h))

# 按照轮廓中x的位置将轮廓从左到右排序
locs = sorted(locs, key=lambda x: x[0])

# 存储最终匹配出的数字信息
output = []

# 遍历每一个轮廓中的数字
# i代表索引值
for (i, (gX, gY, gW, gH)) in enumerate(locs):
    # 存储每一组匹配后的数字信息
    groupOutput = []

    # 根据坐标提取每一个组
    group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
    cv_show('group', group)
    # 预处理
    # 二值处理
    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv_show('group', group)
    # 计算每一组的轮廓
    digitCnts, hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 将每一组中的轮廓进行排序
    digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]

    # 计算每一组中的每一个数值
    for c in digitCnts:
        # 找到当前数值的轮廓，resize成合适的的大小
        (x, y, w, h) = cv2.boundingRect(c)
        roi = group[y:y + h, x:x + w]
        # 这个地方是待检测目标，待检测目标的大小需要缩放和模板一样的大小
        roi = cv2.resize(roi, (57, 88))
        cv_show('roi', roi)

        # 使用模板开始验证

        # 计算匹配得分
        scores = []
        # 在模板中计算每一个得分（digit代表每个模板匹配的数字，digitROI是对应数字的模板）
        for (digit, digitROI) in digits.items():
            # 模板匹配（roi: 检测目标，digitROI：模板，cv2.TM_CCOEFF: 匹配规则（计算相关系数，计算出来的值越大，越相关））
            result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)

        # 根据匹配规则，使用合适的算法得到最合适的数字
        # 比如这个示例中使用的是cv2.TM_CCOEFF匹配规则，scores越大代表越相关
        groupOutput.append(str(np.argmax(scores)))

    # 在原始图像中绘制矩形
    cv2.rectangle(image, (gX - 5, gY - 5), (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
    # 在原始图像中加上文字提示
    cv2.putText(image, "".join(groupOutput), (gX, gY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    # 得到结果
    output.extend(groupOutput)

# 打印结果
print("Credit Card Type: {}".format(FIRST_NUMBER[output[0]]))
print("Credit Card #: {}".format("".join(output)))
cv2.imshow("Image", image)
cv2.waitKey(0)