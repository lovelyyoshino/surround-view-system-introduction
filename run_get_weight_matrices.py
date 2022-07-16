import os
import numpy as np
import cv2
from PIL import Image
from surround_view import FisheyeCameraModel, display_image, BirdView
import surround_view.param_settings as settings


def main():
    names = settings.camera_names#获取相机名称
    images = [os.path.join(os.getcwd(), "images", name + ".png") for name in names]#获取相机的图像文件
    yamls = [os.path.join(os.getcwd(), "yaml", name + ".yaml") for name in names]#获取相机的yaml文件
    camera_models = [FisheyeCameraModel(camera_file, camera_name) for camera_file, camera_name in zip (yamls, names)]#鱼眼镜头处理方式完成ipm处理

    projected = []
    for image_file, camera in zip(images, camera_models):#获取相机的图像文件和相机的参数
        img = cv2.imread(image_file)#读取图像
        img = camera.undistort(img)#去畸变图像
        img = camera.project(img)#投影图像
        img = camera.flip(img)#翻转图像
        projected.append(img)#投影图像

    birdview = BirdView()
    Gmat, Mmat = birdview.get_weights_and_masks(projected)#获取权重并求得掩膜
    birdview.update_frames(projected)#更新图像
    birdview.make_luminance_balance().stitch_all_parts()#获取拼接图像
    birdview.make_white_balance()#获取白平衡图像
    birdview.copy_car_image()
    ret = display_image("BirdView Result", birdview.image)
    if ret > 0:
        Image.fromarray((Gmat * 255).astype(np.uint8)).save("weights.png")
        Image.fromarray(Mmat.astype(np.uint8)).save("masks.png")


if __name__ == "__main__":
    main()
