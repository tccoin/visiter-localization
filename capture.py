import cv2
import pymynt
import helper
import numpy as np
import time

if __name__ == "__main__":
    count = 0
    start_capture = False
    pymynt.init_camera('raw')
    while True:
        tic = time.time()
        depth = pymynt.get_depth_image()
        if depth.shape[0] < 10:
            continue
        left = pymynt.get_left_image()
        cv2.imshow('left',left)
        right = pymynt.get_right_image()
        # cv2.imshow('right',right)
        if left.shape[0] < 10 or right.shape[0] < 10:
            continue
        # np.save('data/mynt/depth_{}.npy'.format(count),depth)
        if start_capture:
            count = count+1
            cv2.imwrite('data/mynt/depth_{}.png'.format(count),depth.astype(np.uint16))
            cv2.imwrite('data/mynt/left_{}.jpg'.format(count),left)
            cv2.imwrite('data/mynt/right_{}.jpg'.format(count),right)
            # key=cv2.waitKey(1)
            toc = time.time()
            print('Saving: {}  fps: {:.1f}'.format(count, 1/(toc-tic)))
        else:
            key=cv2.waitKey(1)
            if key==32:
                start_capture=True
            toc = time.time()
            print('fps: {:.1f}'.format(1/(toc-tic)))