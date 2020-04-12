import cv2
import numpy as np

if __name__ == "__main__":
    for i in range(1,100):
        depth = cv2.imread('data/mynt/depth_{}.png'.format(i), cv2.IMREAD_UNCHANGED)
        # print(depth.dtype)
        depth_mat = depth/np.max(depth)*255
        depth_mat = depth_mat.astype('uint8')
        depth_mat = cv2.cvtColor(depth_mat, cv2.COLOR_GRAY2RGB)
        cv2.imshow('depth',depth_mat)
        cv2.waitKey(100)