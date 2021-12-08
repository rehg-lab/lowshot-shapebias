import os
import cv2
import numpy as np
import pdb
import argparse

from joblib import Parallel, delayed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='path to dataset to overlay backgrounds to images, for example the absolute path where Blender rendered the data')
    args = parser.parse_args()

    img_file_paths = []
    for dirname, subdirs, files in os.walk(args.data_path):
        for fname in files:
            fpath = os.path.join(dirname, fname)
            if 'image_output' in fpath or 'segmentation_output' in fpath or 'depth_output' in fpath:
                img_file_paths.append(fpath)

    results = Parallel(n_jobs=12, verbose=1, backend="multiprocessing")(map(delayed(job), img_file_paths))

def transparentOverlay(overlay,src):
    """
    :param src: Input Color Background Image
    :param overlay: transparent Image (BGRA)
    """
    h,w,_ = overlay.shape  # Size of foreground
    rows,cols,_ = src.shape  # Size of background Image

    alpha = overlay[:,:,3]/255.0
    src[:,:,0] = np.multiply(alpha,overlay[:,:,0]) + np.multiply((1-alpha),src[:,:,0])
    src[:,:,1] = np.multiply(alpha,overlay[:,:,1]) + np.multiply((1-alpha),src[:,:,1])
    src[:,:,2] = np.multiply(alpha,overlay[:,:,2]) + np.multiply((1-alpha),src[:,:,2])

    return src

def job(img_filepath):

    try:
        foreground = cv2.imread(img_filepath, cv2.IMREAD_UNCHANGED)
        f_h, f_w, _ = foreground.shape
    except:
        print(img_filepath + " could not be read")
        return 0

    if foreground.shape[2] != 4:
        return 0

    if 'image_output' in img_filepath:
        # white bg
        background = np.ones((f_h, f_w, 3))*255

    if 'depth_output' in img_filepath:
        # black bg
        background = np.zeros((f_h, f_w, 3))
    
    if 'segmentation_output' in img_filepath:
        # black bg
        background = np.zeros((f_h, f_w, 3))

    img = transparentOverlay(foreground, background)
    cv2.imwrite(img_filepath, img)

    return 1

if __name__ == "__main__":
    main()
