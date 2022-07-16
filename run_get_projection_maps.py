"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Manually select points to get the projection map
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import argparse
import os
import numpy as np
import cv2
from surround_view import FisheyeCameraModel, PointSelector, display_image
import surround_view.param_settings as settings


def get_projection_map(camera_model, image):#获取投影矩阵
    und_image = camera_model.undistort(image)#去畸变图像
    name = camera_model.camera_name#相机名称
    gui = PointSelector(und_image, title=name)#获取去畸变图像的四个点
    dst_points = settings.project_keypoints[name]#获取相机的四个点
    choice = gui.loop()
    if choice > 0:
        src = np.float32(gui.keypoints)
        dst = np.float32(dst_points)
        camera_model.project_matrix = cv2.getPerspectiveTransform(src, dst)#获得转换的点
        proj_image = camera_model.project(und_image)#投影图像

        ret = display_image("Bird's View", proj_image)#显示投影图像
        if ret > 0:
            return True
        if ret < 0:
            cv2.destroyAllWindows()

    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-camera", required=True,
                        choices=["front", "back", "left", "right"],
                        help="The camera view to be projected")
    parser.add_argument("-scale", nargs="+", default=None,
                        help="scale the undistorted image")
    parser.add_argument("-shift", nargs="+", default=None,
                        help="shift the undistorted image")
    args = parser.parse_args()

    if args.scale is not None:
        scale = [float(x) for x in args.scale]
    else:
        scale = (1.0, 1.0)

    if args.shift is not None:
        shift = [float(x) for x in args.shift]
    else:
        shift = (0, 0)

    camera_name = args.camera#获取相机名称
    camera_file = os.path.join(os.getcwd(), "yaml", camera_name + ".yaml")#获取相机的yaml文件
    image_file = os.path.join(os.getcwd(), "images", camera_name + ".png")#获取相机的图像文件
    image = cv2.imread(image_file)#读取图像
    camera = FisheyeCameraModel(camera_file, camera_name)#获取相机的参数
    camera.set_scale_and_shift(scale, shift)#设置相机的参数
    success = get_projection_map(camera, image)#获取投影矩阵
    if success:
        print("saving projection matrix to yaml")
        camera.save_data()
    else:
        print("failed to compute the projection map")


if __name__ == "__main__":
    main()
