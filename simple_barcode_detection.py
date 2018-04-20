import numpy as np
import cv2
import os
import math


def createLineIterator(P1, P2, img):
    """
    Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

    Parameters:
        -P1: a numpy array that consists of the coordinate of the first point (x,y)
        -P2: a numpy array that consists of the coordinate of the second point (x,y)
        -img: the image being processed

    Returns:
        -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])
    """
    # define local variables for readability

    imageH = img.shape[0]
    imageW = img.shape[1]
    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]

    # difference and absolute difference between points
    # used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    # predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dYa, dXa), 3), dtype=np.float32)
    itbuffer.fill(np.nan)

    # Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X:  # vertical line segment
        itbuffer[:, 0] = P1X
        if negY:
            itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
        else:
            itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
    elif P1Y == P2Y:  # horizontal line segment
        itbuffer[:, 1] = P1Y
        if negX:
            itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
        else:
            itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
    else:  # diagonal line segment
        steepSlope = dYa > dXa
        if steepSlope:
            slope = dX.astype(np.float32) / dY.astype(np.float32)
            if negY:
                itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
            else:
                itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
            itbuffer[:, 0] = (slope * (itbuffer[:, 1] - P1Y)).astype(np.int) + P1X
        else:
            slope = dY.astype(np.float32) / dX.astype(np.float32)
            if negX:
                itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
            else:
                itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
            itbuffer[:, 1] = (slope * (itbuffer[:, 0] - P1X)).astype(np.int) + P1Y

    # Remove points outside of image
    colX = itbuffer[:, 0]
    colY = itbuffer[:, 1]
    itbuffer = itbuffer[(colX >= 0) & (colY >= 0) & (colX < imageW) & (colY < imageH)]

    # Get intensities from img ndarray
    itbuffer[:, 2] = img[itbuffer[:, 1].astype(np.uint), itbuffer[:, 0].astype(np.uint)]

    return itbuffer

def cv_distance(P, Q):
    return int(math.sqrt(pow((P[0] - Q[0]), 2) + pow((P[1] - Q[1]), 2)))

def isTimingPattern(line):
    # 除去开头结尾的白色像素点
    while line[0] != 0:
        line = line[1:]
    while line[-1] != 0:
        line = line[:-1]
    # 计数连续的黑白像素点
    c = []
    count = 1
    l = line[0]
    for p in line[1:]:
        if p == l:
            count = count + 1
        else:
            c.append(count)
            count = 1
        l = p
    c.append(count)
    # 如果黑白间隔太少，直接排除
    if len(c) < 5:
        return False
    # 计算方差，根据离散程度判断是否是 Timing Pattern
    threshold = 5
    result = np.var(c)
    return result < threshold

def detect(image):
    # 把图像从RGB装换成灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.GaussianBlur(gray,(5,5),0)
    edges = cv2.Canny(gray, 100, 200)
    # cv2.imshow('img',edges)
    # cv2.waitKey(0)
    img_fc, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    hierarchy = hierarchy[0]
    found = []
    for i in range(len(contours)):
        k = i
        c = 0
        while hierarchy[k][2] != -1:
            k = hierarchy[k][2]
            c = c + 1
        if c >= 5:
            found.append(i)

    # for i in found:
    #     img_dc = image.copy()
    #     cv2.drawContours(img_dc, contours, i, (0, 255, 0), 3)
    #     cv2.imshow('img',img_dc)
    #     cv2.waitKey(0)
    # draw_img = image.copy()
    # for i in found:
    #     rect = cv2.minAreaRect(contours[i])
    #     box = cv2.boxPoints(rect)
    #     box = np.int0(box)
    #     cv2.drawContours(draw_img, [box], 0, (0, 0, 255), 2)
    # cv2.imshow('img',draw_img)
    # cv2.waitKey(0)
    boxes = []
    for i in found:
        # print(contours[i])
        rect = cv2.minAreaRect(contours[i])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # box = map(tuple, box)
        box = tuple(box)
        boxes.append(box)



    def check(a, b):
        # 存储 ab 数组里最短的两点的组合
        s1_ab = ()
        s2_ab = ()
        # 存储 ab 数组里最短的两点的距离，用于比较
        s1 = np.iinfo('i').max
        s2 = s1
        for ai in a:
            for bi in b:
                d = cv_distance(ai, bi)
                if d < s2:
                    if d < s1:
                        s1_ab, s2_ab = (ai, bi), s1_ab
                        s1, s2 = d, s1
                    else:
                        s2_ab = (ai, bi)
                        s2 = d
        a1, a2 = s1_ab[0], s2_ab[0]
        b1, b2 = s1_ab[1], s2_ab[1]
        # a1 = tuple(a1)
        # b1 = tuple(b1)
        # a2 = tuple(a2)
        # b2 = tuple(b2)
        # 将最短的两个线画出来
        # cv2.line(draw_img, tuple(a1), tuple(b1), (0, 0, 255), 3)
        # cv2.line(draw_img, tuple(a2), tuple(b2), (0, 0, 255), 3)

        th, bi_img = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        a1 = (a1[0] + (a2[0] - a1[0]) * 1 / 14, a1[1] + (a2[1] - a1[1]) * 1 / 14)
        b1 = (b1[0] + (b2[0] - b1[0]) * 1 / 14, b1[1] + (b2[1] - b1[1]) * 1 / 14)
        # a2 = np.array([a2[0] + (a1[0] - a2[0]) * 1 / 14, a2[1] + (a1[1] - a2[1]) * 1 / 14])
        # b2 = np.array([b2[0] + (b1[0] - b2[0]) * 1 / 14, b2[1] + (b1[1] - b2[1]) * 1 / 14])

        buffer = createLineIterator(a1,b1,bi_img)
        buffer = [b[2] for b in buffer]
        # print(buffer)
        return isTimingPattern(buffer)


    valid = set()
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            # check(boxes[i], boxes[j])
            # if check(boxes[i], boxes[j]):
            if len(box)>3:
                valid.add(i)
                valid.add(j)

    valid = list(valid)
    if len(valid)>2:
        contour_all = list()
        for i in range(len(valid)):
            c = found[valid.pop()]
            for su in contours[c]:
                contour_all.append(su)
        contour_all= np.array(contour_all)
        rect = cv2.minAreaRect(contour_all)
        box = cv2.boxPoints(rect)
        box = np.array(box)
        # print(box)
        # draw_img = image.copy()
        # cv2.polylines(draw_img, np.int32([box]), True, (0, 0, 255), 10)
        # cv2.imshow('img', draw_img)
        # cv2.waitKey(0)
        return box
    else:
        return None


    # th, bi_img = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    # a1 = (a1[0] + (a2[0] - a1[0]) * 1 / 14, a1[1] + (a2[1] - a1[1]) * 1 / 14)
    # b1 = (b1[0] + (b2[0] - b1[0]) * 1 / 14, b1[1] + (b2[1] - b1[1]) * 1 / 14)
    # a2 = (a2[0] + (a1[0] - a2[0]) * 1 / 14, a2[1] + (a1[1] - a2[1]) * 1 / 14)
    # b2 = (b2[0] + (b1[0] - b2[0]) * 1 / 14, b2[1] + (b1[1] - b2[1]) * 1 / 14)
    #使用Scharr操作（指定使用ksize = -1）构造灰度图在水平和竖直方向上的梯度幅值表示。
    # gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    # gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

    # Scharr操作后，从x的梯度减去y的梯度
    # gradient = cv2.subtract(gradX, gradY)
    # gradient = cv2.convertScaleAbs(gradient)

    # 对上述的梯度图采用用9x9的核进行平均模糊,这是有利于降噪的
    # 然后进行二值化处理，要么是255(白)要么是0(黑)
    # blurred = cv2.blur(gradient, (9, 9))
    # (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

    # construct a closing kernel and apply it to the thresholded image
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    # closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # perform a series of erosions and dilations
    # closed = cv2.erode(closed, None, iterations=4)
    # closed = cv2.dilate(closed, None, iterations=4)

    # find the contours in the thresholded image
    # binary, cnts, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # # cv2.drawContours(image, cnts, -1, (0, 0, 255), 3)
    # # if no contours were found, return None
    # if len(cnts) == 0:
    #     return None
    # # cv2.imshow("0", binary)
    # # cv2.waitKey(0)
    #     # otherwise, sort the contours by area and compute the rotated
    # # bounding box of the largest contour
    # c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    # rect = cv2.minAreaRect(c)
    # box = np.int0(cv2.boxPoints(rect))
    # # box[0] = box[0]
    # # box[1] = box[1] -10
    # # box[2] = box[2]
    # # box[3] = box[3] + 10
    # print(box)
    # # cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
    # # cv2.imshow("Image", image)
    # # cv2.imwrite("contoursImage2.jpg", image)
    # # cv2.waitKey(0)
    # # cv2.imwrite(r"C:\Users\User\Desktop\demo\0.jpg",image)
    # # Xs = [i[0] for i in box]
    # # Ys = [i[1] for i in box]
    # # x1 = min(Xs)
    # # x2 = max(Xs)
    # # y1 = min(Ys)
    # # y2 = max(Ys)
    # # hight = y2 - y1
    # # width = x2 - x1
    # # cropImg = image[y1:y1 + hight, x1:x1 + width]
    # # cv2.imshow("Image", cropImg)
    # # cv2.imwrite("copyImg.jpg", cropImg)
    # # cv2.waitKey(0)
    # # return the bounding box of the barcode
    # # cropImg = cv2.cvtColor(cropImg, cv2.COLOR_BGR2GRAY)
    # return box

def open_image():
    img = cv2.imread(r"C:\Users\User\Desktop\demo\0.jpg")
    # img = cv2.resize(img,(399,260))
    if img is not None:
        cropImg = detect(img)
        # cv2.drawContours(img, [box], -1, (0, 255, 0), 3)
        # cv2.imshow("Image", cropImg)
        # cv2.imwrite("contoursImage2.jpg", img)
        # cv2.waitKey(0)
    else:
        print("图片无法打开！")


if __name__ == '__main__':
    open_image()