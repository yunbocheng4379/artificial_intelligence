# 导入工具包
import numpy as np
import argparse
import cv2

# 设置参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image to be scanned") 
args = vars(ap.parse_args())

def order_points(pts):
	# 一共4个坐标点
	rect = np.zeros((4, 2), dtype = "float32")

	# 按顺序找到对应坐标0123分别是 左上，右上，右下，左下
	# 计算左上，右下
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# 计算右上和左下
	diff = np.diff(pts, axis = 1)
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
		[0, maxHeight - 1]], dtype = "float32")

	# 透视变换：将图像中倾斜或扭曲区域（如文档、车牌等）矫正为正面视图。
	# 计算从源坐标 rect 到目标坐标 dst 的 透视变换矩阵 M
	M = cv2.getPerspectiveTransform(rect, dst)
	# 应用矩阵 M 对图像进行透视变换，生成矫正后的图像 warped
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	# 返回变换后结果
	return warped

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
	dim = None
	(h, w) = image.shape[:2]
	if width is None and height is None:
		return image
	if width is None:
		r = height / float(h)
		dim = (int(w * r), height)
	else:
		r = width / float(w)
		dim = (width, int(h * r))
	resized = cv2.resize(image, dim, interpolation=inter)
	return resized

# 读取输入
image = cv2.imread(args["image"])
# 保留一个缩放的比例，便于最终将图像变换回来，缩放后相应坐标也会相同变化
ratio = image.shape[0] / 500.0
orig = image.copy()
# 重置图像大小
image = resize(orig, height = 500)

# 预处理
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 高斯滤波（平滑处理），抑制图像噪音、平滑细节。
gray = cv2.GaussianBlur(gray, (5, 5), 0)
# 边缘检测
edged = cv2.Canny(gray, 75, 200)

# 展示预处理结果
print("STEP 1: 边缘检测")
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 轮廓检测
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
# cv2.contourArea计算面积
# 按照面积进行排序，并按照面积排序后取前五个最大的轮廓
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

# 遍历轮廓
for c in cnts:
	# 计算轮廓周长，True表示闭合
	peri = cv2.arcLength(c, True)

	# C表示输入的点集
	# epsilon表示从原始轮廓到近似轮廓的最大距离，它是一个准确度参数
	# True表示封闭的
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)

	# 经过上述的变换，将轮廓转换成为一个矩形，最终会保存矩形的四个顶点
	if len(approx) == 4:
		screenCnt = approx
		break

# 展示结果
print("STEP 2: 获取轮廓")
# 绘制轮廓：-1参数代表将screenCnt中的轮廓全部绘制
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 透视变换：将图像中倾斜或扭曲区域（如文档、车牌等）矫正为正面视图
# screenCnt.reshape(4, 2) * ratio： screenCnt计算出来的点位是图像经过resize之后的
# screenCnt.reshape(4, 2)：将轮廓点转换为标准的坐标矩阵形式
# 现在要在原始图像中绘制出这个位置，需要将点位进行还原。
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
# 二值处理
ref = cv2.threshold(warped, 100, 255, cv2.THRESH_BINARY)[1]
# cv2.imwrite是OpenCV中的图像保存函数，将图像数据（存储在ref变量中）以JPEG格式写入到scan.jpg中。
cv2.imwrite('scan.jpg', ref)

# 展示结果
print("STEP 3: 变换")
cv2.imshow("Original", resize(orig, height = 650))
cv2.imshow("Scanned", resize(ref, height = 650))
cv2.waitKey(0)