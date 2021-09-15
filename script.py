import cv2
import numpy as np
import spectral.io.envi as envi
from spectral import *

#SWIR=cv2.imread("SWIR_gausian_enhancement_990_1010.jpg",0)
#VNIR=cv2.imread("VNIR_gausian_enhancement_970_990.jpg",0)

#window_size=5
#lmbda=8000
#sigma=1.5

#left_matcher=cv2.StereoSGBM_create(minDisparity=0,numDisparities=64,blockSize=9, 
#P1=8 * 3 * window_size ** 2,P2=32 * 3 * window_size ** 2, 
#disp12MaxDiff=1,uniquenessRatio=1,speckleWindowSize=0,speckleRange=1,
#preFilterCap=1,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

#right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

#wls_filter=cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
#wls_filter.setLambda(lmbda)
#wls_filter.setSigmaColor(sigma)

#displ = left_matcher.compute(SWIR, VNIR)  # .astype(np.float32)/16
#dispr = right_matcher.compute(VNIR, SWIR)  # .astype(np.float32)/16
#displ = np.int16(displ)
#dispr = np.int16(dispr)

#filteredImg = wls_filter.filter(displ, SWIR, None, dispr)  # important to put "imgL" here!!!

#cv2.imwrite(r'C:\Users\andre\Desktop\PythonTest\disparity_filtered_P2_32_9_w3_gaus.jpg',filteredImg)
#disparity=cv2.imread("test2.jpg",0)


#pcl = bib.pointcloud(disparity, 20)
#np.savetxt('recon.xyz', pcl)

#pcl2=bib.matrixCalc(disparity,20)
#np.savetxt('recon2.xyz', pcl2)
##################################

img = envi.open('nmbu2_07_Mjolnir_S620_SN7003_raw_rad_keystone_smile_bsq_float32.hdr', 'nmbu2_07_Mjolnir_S620_SN7003_raw_rad_keystone_smile_bsq_float32.img').read_band(0) 
view=imshow(img)


input("press enter to cont")
print("All done!")