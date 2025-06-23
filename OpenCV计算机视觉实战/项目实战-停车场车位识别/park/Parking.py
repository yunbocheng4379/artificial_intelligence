import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


class Parking:

    def show_images(self, images, cmap=None):
        cols = 2
        rows = (len(images) + 1) // cols

        plt.figure(figsize=(15, 12))
        for i, image in enumerate(images):
            plt.subplot(rows, cols, i + 1)
            cmap = 'gray' if len(image.shape) == 2 else cmap
            plt.imshow(image, cmap=cmap)
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout(pad=0, h_pad=0, w_pad=0)
        plt.show()

    def cv_show(self, name, img):
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def select_rgb_white_yellow(self, image):
        # 过滤掉背景，基于颜色阈值从图像中提取白色及亮色区域
        # 颜色阈值定义
        # 目标颜色：筛选出BGR颜色空间中各通道值均在[120,255]的像素，即白色或近白色的亮色区域（如浅黄）
        # BGR最低阈值
        lower = np.uint8([120, 120, 120])
        # BGR最高阈值
        upper = np.uint8([255, 255, 255])

        # 生成二值（黑白）掩膜（Mask），
        # 低于lower_red和高于upper_red的部分分别变成0，lower_red～upper_red之间的值变成255,相当于过滤背景
        # 创建mask，这个mask的尺寸必须和覆盖的原始图像尺寸一致
        # 淹膜含义：白色区域表示符合阈值范围的目标颜色，黑色区域为背景
        white_mask = cv2.inRange(image, lower, upper)
        self.cv_show('white_mask', white_mask)

        # 应用掩膜保留目标区域
        # 作用：将原图与掩膜按位与运算，仅保留掩膜白色区域的原始颜色，其余部分变黑。
        masked = cv2.bitwise_and(image, image, mask=white_mask)
        self.cv_show('masked', masked)
        return masked

    def convert_gray_scale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def detect_edges(self, image, low_threshold=50, high_threshold=200):
        return cv2.Canny(image, low_threshold, high_threshold)

    def filter_region(self, image, vertices):
        """
                通过多边形掩膜从图像中提取特定区域，并剔除掉区域外的无关部分。
        """
        # 创建一个与输入图像尺寸和数据类型完全相同的全零数组（黑色图像），用于后续作为掩膜（Mask）的基础。
        # 若image是彩色图（BGR格式，形状为（H,W,3）），则mask形状也是（H,W,3），则mask为三通道全零矩阵。
        # 若image是灰度图（形状为（H,W）），则mask也为单通道全零矩阵。
        mask = np.zeros_like(image)

        # 判断条件：仅当输入为灰度图时，向掩膜填充多边形区域。
        if len(mask.shape) == 2:
            # vertices: 多边形顶点坐标数组（格式为 [[[x1,y1], [x2,y2], ...]]）。
            # cv2.fillPoly：将多边形内的像素设为白色（255），其余保持黑色（0）。
            cv2.fillPoly(mask, vertices, 255)
            self.cv_show('mask', mask)
        # 应用掩膜保留目标区域：将原图与掩膜按位与操作，仅保留掩膜白色区域的图像信息，其余变黑。
        return cv2.bitwise_and(image, mask)

    def select_region(self, image):
        """
                手动选择区域
        """
        # first, define the polygon by vertices
        rows, cols = image.shape[:2]
        pt_1 = [cols * 0.05, rows * 0.90]
        pt_2 = [cols * 0.05, rows * 0.70]
        pt_3 = [cols * 0.30, rows * 0.55]
        pt_4 = [cols * 0.6, rows * 0.15]
        pt_5 = [cols * 0.90, rows * 0.15]
        pt_6 = [cols * 0.90, rows * 0.90]

        vertices = np.array([[pt_1, pt_2, pt_3, pt_4, pt_5, pt_6]], dtype=np.int32)
        point_img = image.copy()
        # 绘制时将灰度图转换为RGB图进行绘制
        point_img = cv2.cvtColor(point_img, cv2.COLOR_GRAY2RGB)
        # 使用红色的圆圈将这些点位进行绘制
        for point in vertices[0]:
            cv2.circle(point_img, (point[0], point[1]), 10, (0, 0, 255), 4)
        self.cv_show('point_img', point_img)

        return self.filter_region(image, vertices)

    def hough_lines(self, image):
        # 输入的图像需要是边缘检测后的结果
        # minLineLength(线的最短长度，比这个短的都被忽略)和MaxLineCap（两条直线之间的最大间隔，小于此值，认为是一条直线）。
        # rho距离精度,theta角度精度, threshold超过设定阈值才被检测出线段。
        # hough_lines 方法通过 概率霍夫变换（HoughLinesP） 在边缘检测后的图像中检测直线段。
        # 在边缘检测后的图像中，将不连续的边缘像素点组合程近似直线段（或特定形状）的数学模型。
        return cv2.HoughLinesP(image, rho=0.1, theta=np.pi / 10, threshold=15, minLineLength=9, maxLineGap=4)

    def draw_lines(self, image, lines, color=[255, 0, 0], thickness=2, make_copy=True):
        # 过滤霍夫变换检测到直线（只保留水平直线）
        if make_copy:
            image = np.copy(image)
        cleaned = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                # 绘制每条线段至少需要两个点(x1,y1)和(x2,y2)
                # abs(y2 - y1) <= 1：确定线段方向必须是水平的，两点高度不能超过1
                # abs(x2 - x1) >= 25 and abs(x2 - x1) <= 55：确定线段长度必须在 25~55 之间
                if abs(y2 - y1) <= 1 and 25 <= abs(x2 - x1) <= 55:
                    cleaned.append((x1, y1, x2, y2))
                    # 绘制
                    # image：在哪个图像上绘制
                    # (x1,y1),(x2,y2)：绘制直线的两个坐标点
                    # color：直线颜色
                    # thickness：直线宽度，单位为像素
                    cv2.line(image, (x1, y1), (x2, y2), color, thickness)
        print(" No lines detected: ", len(cleaned))
        # 返回绘制上直线线段的图片
        return image

    def identify_blocks(self, image, lines, make_copy=True):
        if make_copy:
            new_image = np.copy(image)
        # Step 1: 过滤部分直线
        # 目的：只保留符合停车线的直线（水平、长度）
        cleaned = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                if abs(y2 - y1) <= 1 and 25 <= abs(x2 - x1) <= 55:
                    cleaned.append((x1, y1, x2, y2))

        # Step 2: 对直线按照x1进行排序
        # 目的：将各个直线按照x1坐标从小到大排序，为分组做准备
        import operator
        list1 = sorted(cleaned, key=operator.itemgetter(0, 1))

        # Step 3: 找到多个列，相当于每列是一排车
        # 目的：将每列按照 {0: 第一列直线集合,1: 第二列直线集合...}格式存储
        clusters = {}
        dIndex = 0
        # 确定每列最小间隔，小于这个范围认为是同一列直线，大于这个范围不是同一列直线
        clus_dist = 10

        for i in range(len(list1) - 1):
            distance = abs(list1[i + 1][0] - list1[i][0])
            if distance <= clus_dist:
                if not dIndex in clusters.keys(): clusters[dIndex] = []
                clusters[dIndex].append(list1[i])
                clusters[dIndex].append(list1[i + 1])
            else:
                dIndex += 1

        # Step 4: 得到坐标
        # 目的：计算每列划分的矩形坐标值
        rects = {}
        i = 0
        for key in clusters:
            all_list = clusters[key]
            cleaned = list(set(all_list))
            if len(cleaned) > 5:
                cleaned = sorted(cleaned, key=lambda tup: tup[1])
                avg_y1 = cleaned[0][1]
                avg_y2 = cleaned[-1][1]
                avg_x1 = 0
                avg_x2 = 0
                for tup in cleaned:
                    avg_x1 += tup[0]
                    avg_x2 += tup[2]
                avg_x1 = avg_x1 / len(cleaned)
                avg_x2 = avg_x2 / len(cleaned)
                rects[i] = (avg_x1, avg_y1, avg_x2, avg_y2)
                i += 1
        print("Num Parking Lanes: ", len(rects))

        # Step 5: 把列矩形画出来
        # 目的：在原始图像中绘制每列划分的矩形
        buff = 7
        for key in rects:
            tup_topLeft = (int(rects[key][0] - buff), int(rects[key][1]))
            tup_botRight = (int(rects[key][2] + buff), int(rects[key][3]))
            cv2.rectangle(new_image, tup_topLeft, tup_botRight, (0, 255, 0), 3)
        return new_image, rects

    def draw_parking(self, image, rects, make_copy=True, color=[255, 0, 0], thickness=2, save=True):
        if make_copy:
            new_image = np.copy(image)
        gap = 15.5
        spot_dict = {}  # 字典：一个车位对应一个位置
        tot_spots = 0
        # 微调
        adj_y1 = {0: 20, 1: -10, 2: 0, 3: -11, 4: 28, 5: 5, 6: -15, 7: -15, 8: -10, 9: -30, 10: 9, 11: -32}
        adj_y2 = {0: 30, 1: 50, 2: 15, 3: 10, 4: -15, 5: 15, 6: 15, 7: -20, 8: 15, 9: 15, 10: 0, 11: 30}

        adj_x1 = {0: -8, 1: -15, 2: -15, 3: -15, 4: -15, 5: -15, 6: -15, 7: -15, 8: -10, 9: -10, 10: -10, 11: 0}
        adj_x2 = {0: 0, 1: 15, 2: 15, 3: 15, 4: 15, 5: 15, 6: 15, 7: 15, 8: 10, 9: 10, 10: 10, 11: 0}
        for key in rects:
            tup = rects[key]
            x1 = int(tup[0] + adj_x1[key])
            x2 = int(tup[2] + adj_x2[key])
            y1 = int(tup[1] + adj_y1[key])
            y2 = int(tup[3] + adj_y2[key])
            cv2.rectangle(new_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            num_splits = int(abs(y2 - y1) // gap)
            for i in range(0, num_splits + 1):
                y = int(y1 + i * gap)
                cv2.line(new_image, (x1, y), (x2, y), color, thickness)
            if key > 0 and key < len(rects) - 1:
                # 竖直线
                x = int((x1 + x2) / 2)
                cv2.line(new_image, (x, y1), (x, y2), color, thickness)
            # 计算数量
            if key == 0 or key == (len(rects) - 1):
                tot_spots += num_splits + 1
            else:
                tot_spots += 2 * (num_splits + 1)

            # 字典对应好
            if key == 0 or key == (len(rects) - 1):
                for i in range(0, num_splits + 1):
                    cur_len = len(spot_dict)
                    y = int(y1 + i * gap)
                    spot_dict[(x1, y, x2, y + gap)] = cur_len + 1
            else:
                for i in range(0, num_splits + 1):
                    cur_len = len(spot_dict)
                    y = int(y1 + i * gap)
                    x = int((x1 + x2) / 2)
                    spot_dict[(x1, y, x, y + gap)] = cur_len + 1
                    spot_dict[(x, y, x2, y + gap)] = cur_len + 2

        print("total parking spaces: ", tot_spots, cur_len)
        if save:
            # 保存图像信息到with_parking.jpg
            filename = 'with_parking.jpg'
            cv2.imwrite(filename, new_image)
        # new_image：处理后的图像（标注了车位边界、占用状态等）
        # spot_dict：一个字典，存储每个停车位坐标
        return new_image, spot_dict

    def assign_spots_map(self, image, spot_dict, make_copy=True, color=[255, 0, 0], thickness=2):
        if make_copy:
            new_image = np.copy(image)
        for spot in spot_dict.keys():
            (x1, y1, x2, y2) = spot
            cv2.rectangle(new_image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        return new_image

    def save_images_for_cnn(self, image, spot_dict, folder_name='cnn_data'):
        try:
            for spot in spot_dict.keys():
                # 获取每个停车位坐标
                (x1, y1, x2, y2) = spot
                (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
                # 在原始图片按照点位裁剪（裁剪的图片中存在汽车信息）
                spot_img = image[y1:y2, x1:x2]
                # 调整大小
                spot_img = cv2.resize(spot_img, (0, 0), fx=2.0, fy=2.0)
                spot_id = spot_dict[spot]
                # 存储裁剪后每个车位信息
                filename = 'spot' + str(spot_id) + '.jpg'
                print(spot_img.shape, filename, (x1, x2, y1, y2))

                cv2.imwrite(os.path.join(folder_name, filename), spot_img)
        except Exception as e:
            print(e)

    def make_prediction(self, image, model, class_dictionary):
        # 预处理
        img = image / 255.

        # 转换成4D tensor
        image = np.expand_dims(img, axis=0)

        # 用训练好的模型进行训练
        class_predicted = model.predict(image)
        inID = np.argmax(class_predicted[0])
        label = class_dictionary[inID]
        return label

    def predict_on_image(self, image, spot_dict, model, class_dictionary, make_copy=True, color=[0, 255, 0], alpha=0.5):
        if make_copy:
            new_image = np.copy(image)
            overlay = np.copy(image)
        self.cv_show('new_image', new_image)
        cnt_empty = 0
        all_spots = 0
        for spot in spot_dict.keys():
            all_spots += 1
            (x1, y1, x2, y2) = spot
            (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
            spot_img = image[y1:y2, x1:x2]
            # 将图片尺寸修改为和训练模型所需尺寸一致的大小
            spot_img = cv2.resize(spot_img, (48, 48))

            label = self.make_prediction(spot_img, model, class_dictionary)
            if label == 'empty':
                cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
                cnt_empty += 1

        cv2.addWeighted(overlay, alpha, new_image, 1 - alpha, 0, new_image)

        cv2.putText(new_image, "Available: %d spots" % cnt_empty, (30, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

        cv2.putText(new_image, "Total: %d spots" % all_spots, (30, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        save = False

        if save:
            filename = 'with_marking.jpg'
            cv2.imwrite(filename, new_image)
        self.cv_show('new_image', new_image)

        return new_image

    def predict_on_video(self, video_name, final_spot_dict, model, class_dictionary, ret=True):
        """
                逐帧解析视频，实时检测每个停车位的占用状态。
        """
        # cv2.VideoCapture(video_name) 是 OpenCV（Open Source Computer Vision Library） 提供的一个方法，主要用于 从视频文件或摄像头捕获视频帧。
        # 视频帧就是一张图片，检测当前时间点的车位占用情况
        cap = cv2.VideoCapture(video_name)
        count = 0
        while ret:
            ret, image = cap.read()
            count += 1
            if count == 5:
                count = 0
                new_image = np.copy(image)
                overlay = np.copy(image)
                cnt_empty = 0
                all_spots = 0
                color = [0, 255, 0]
                alpha = 0.5
                for spot in final_spot_dict.keys():
                    all_spots += 1
                    (x1, y1, x2, y2) = spot
                    (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
                    spot_img = image[y1:y2, x1:x2]
                    # 将图片尺寸修改为和训练模型所需尺寸一致的大小
                    spot_img = cv2.resize(spot_img, (48, 48))

                    label = self.make_prediction(spot_img, model, class_dictionary)
                    if label == 'empty':
                        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
                        cnt_empty += 1

                cv2.addWeighted(overlay, alpha, new_image, 1 - alpha, 0, new_image)

                cv2.putText(new_image, "Available: %d spots" % cnt_empty, (30, 95),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2)

                cv2.putText(new_image, "Total: %d spots" % all_spots, (30, 125),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2)
                cv2.imshow('frame', new_image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()
        cap.release()
