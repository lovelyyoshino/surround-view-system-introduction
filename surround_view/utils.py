import cv2
import numpy as np


def gstreamer_pipeline(cam_id=0,
                       capture_width=960,
                       capture_height=640,
                       framerate=60,
                       flip_method=2):
    """
    Use libgstreamer to open csi-cameras.
    """
    return ("nvarguscamerasrc sensor-id={} ! ".format(cam_id) + \
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (capture_width,
               capture_height,
               framerate,
               flip_method
            )
    )


def convert_binary_to_bool(mask):
    """
    Convert a binary image (only one channel and pixels are 0 or 255) to
    a bool one (all pixels are 0 or 1).
    """
    return (mask.astype(np.float) / 255.0).astype(np.int)


def adjust_luminance(gray, factor):
    """
    Adjust the luminance of a grayscale image by a factor.
    """
    return np.minimum((gray * factor), 255).astype(np.uint8)


def get_mean_statistisc(gray, mask):
    """
    Get the total values of a gray image in a region defined by a mask matrix.
    The mask matrix must have values either 0 or 1.
    """
    return np.sum(gray * mask)


def mean_luminance_ratio(grayA, grayB, mask):
    return get_mean_statistisc(grayA, mask) / get_mean_statistisc(grayB, mask)


def get_mask(img):
    """
    Convert an image to a mask array.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#转换为灰度图像
    ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)#二值化
    return mask


def get_overlap_region_mask(imA, imB):
    """
    Given two images of the save size, get their overlapping region and
    convert this region to a mask array.
    """
    overlap = cv2.bitwise_and(imA, imB)#获取两个图像的交集
    mask = get_mask(overlap)#获取交集的mask
    mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=2)#膨胀
    return mask


def get_outmost_polygon_boundary(img):
    """
    Given a mask image with the mask describes the overlapping region of
    two images, get the outmost contour of this region.
    """
    mask = get_mask(img)#获取图像的mask
    mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=2)#膨胀
    cnts, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)[-2:]#获取轮廓

    # get the contour with largest aera
    C = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)[0]#获取最大面积的轮廓

    # polygon approximation
    polygon = cv2.approxPolyDP(C, 0.009 * cv2.arcLength(C, True), True)#获取外接矩形

    return polygon


def get_weight_mask_matrix(imA, imB, dist_threshold=5):
    """
    Get the weight matrix G that combines two images imA, imB smoothly.
    """
    overlapMask = get_overlap_region_mask(imA, imB)#获取两个图像的交集
    overlapMaskInv = cv2.bitwise_not(overlapMask)#获取两个图像的交集
    indices = np.where(overlapMask == 255)#获取交集区域

    imA_diff = cv2.bitwise_and(imA, imA, mask=overlapMaskInv)#获取图像A中与交集区域不重合的区域
    imB_diff = cv2.bitwise_and(imB, imB, mask=overlapMaskInv)#获取图像B中与交集区域不重合的区域

    G = get_mask(imA).astype(np.float32) / 255.0#获取图像A的mask，归一化
    # 对重叠区域中的每个像素，利用 `cv2.pointPolygonTest` 计算其到这两个多边形 `polyA` 和 `polyB` 的距离 
    polyA = get_outmost_polygon_boundary(imA_diff)#获取图像A中与交集区域不重合的区域的外接多边形
    polyB = get_outmost_polygon_boundary(imB_diff)#获取图像B中与交集区域不重合的区域的外接多边形

    for y, x in zip(*indices):
        distToB = cv2.pointPolygonTest(polyB, (x, y), True)#获取点到外接多边形的距离
        if distToB < dist_threshold:#如果距离小于阈值，则与A比较
            distToA = cv2.pointPolygonTest(polyA, (x, y), True)
            distToB *= distToB
            distToA *= distToA
            G[y, x] = distToB / (distToA + distToB)#像素对应的权值。即如果这个像素落在 `front` 画面内，则它与 `polyB` 的距离就更远，从而权值更大。

    return G, overlapMask


def make_white_balance(image):
    """
    Adjust white balance of an image base on the means of its channels.
    """
    B, G, R = cv2.split(image)#分离三个通道
    m1 = np.mean(B)#计算均值
    m2 = np.mean(G)
    m3 = np.mean(R)
    K = (m1 + m2 + m3) / 3
    c1 = K / m1#计算系数
    c2 = K / m2
    c3 = K / m3
    B = adjust_luminance(B, c1)#调整亮度
    G = adjust_luminance(G, c2)
    R = adjust_luminance(R, c3)
    return cv2.merge((B, G, R))
