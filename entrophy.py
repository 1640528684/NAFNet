from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import os
# 熵
def entropy(signal):
        lensig=signal.size
        symset=list(set(signal))
        propab=[np.size(signal[signal==i])/(1.0*lensig) for i in symset]#每个值的概率
        ent=np.sum([p*np.log2(1.0/p) for p in propab])
        return ent
if __name__ == '__main__':
    filelist = os.listdir('/data/users/qingluhou/Neural_network/motion_deblur/NAFNet/results/v51-fft-width64-test/visualization/gopro-test_fft')
    for file in filelist:
        if 'gt' in file:
            continue
        # 读图，也可以用Opencv啥的
        colorIm=Image.open(os.path.join('/data/users/qingluhou/Neural_network/motion_deblur/NAFNet/results/v51-fft-width64-test/visualization/gopro-test_fft',file))

        # 灰度
        greyIm=colorIm.convert('L')
        colorIm=np.array(colorIm)
        greyIm=np.array(greyIm)
        N=3
        S=greyIm.shape
        E=np.array(greyIm)


        #以图像左上角为坐标0点
        for row in range(S[0]):
            for col in range(S[1]):
                Left_x=np.max([0,col-N])
                Right_x=np.min([S[1],col+N])
                up_y=np.max([0,row-N])
                down_y=np.min([S[0],row+N])
                region=greyIm[up_y:down_y,Left_x:Right_x].flatten()  # 返回一维数组
                E[row,col]=entropy(region)
        plt.imsave(os.path.join('/data/users/qingluhou/Neural_network/motion_deblur/NAFNet/results/v51-fft-width64-test/visualization/ent',file), E, cmap=plt.cm.jet)
