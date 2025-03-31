import matplotlib.pyplot as plt
import numpy as np
import cv2

image = cv2.imread('C:\\Users\Administrator\Desktop\\new\\V.png', 0)
image = (image - np.mean(image, axis=0)) / np.std(image, axis=0)
colormap = plt.get_cmap('BuPu')
heatmap = (colormap(image) * 2**16).astype(np.uint16)[:,:,:3]
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

# cv2.imshow('image', image)
cv2.imshow('heatmap', heatmap)
cv2.waitKey()
cv2.imwrite('v.png', heatmap)
