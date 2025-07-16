from sewar.full_ref import psnr as psnr_impl
from common.logger import log
from common.path_checker import check_paths
from numpy import float64
import cv2

def psnr(path_img_1: str, path_img_2: str, verbose: bool = False) -> float64 :
    log(path_img_1, path_img_2, verbose)
    check_paths([path_img_1, path_img_2])
    img_1 = cv2.imread(path_img_1)
    img_2 = cv2.imread(path_img_2)
    return psnr_impl(img_1, img_2)

