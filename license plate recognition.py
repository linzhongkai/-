"""
先考虑整体问题的话，车牌识别可以分为以下的部分：
定位问题：用于定位车牌的位置
字符分割问题:用于分割每个字符区域的位置
字符识别问题：用于识别字符

编程风格先按照方法的单元进行
"""

# 导入支持库
import cv2
import numpy as np
from skimage import measure,color,morphology,img_as_ubyte
import matplotlib.pyplot as plt
import pprint

#读取图像
def readimage(imagename,imageflag,showflage):
    image = cv2.imread(imagename,flags=imageflag)
    #使用except异常处理
    #还可以根据读图的内容来进行异常判断，图像内容为none则代表读图失败
    try:
        print("读图成功")
        if(showflage):
            cv2.imshow("image",image)
            print("显示当前图像")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print("关闭当前显示")
        else:
            print("图像不予显示")
    except cv2.error:
        print("读图失败")
    return image



def image_info(image,printflag):
    if(printflag):
        print("IMAGE INFORMATION:")
        print("Image_type: ", type(image)) #图像种类 一般是np数组
        print("Image_dtype", image.dtype) #图像数据类型
        print("Image_shape: ", image.shape) #图像尺寸，不考虑图像的通道数
        print("Image_size: ", image.size) #图像像素点
        # print("Image_pixeldata: ", np.array(image))
    return type(image),image.dtype,image.shape,image.size





# 实现定位
"""
定位问题还需要考虑的是角度的问题
第一步骤暂时实现的是正面角度的定位
此步骤包含图像的处理
"""
def imagelocation(image):
    print("定位操作")
    """
    形态学操作
    """
    print("灰度处理")
    # image_info(image,printflag=True)
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray_image",gray_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print("图像降噪：高斯滤波")
    gb_image = cv2.GaussianBlur(gray_image,(7,7),0)
    # cv2.imshow("gb_image",gb_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print("边缘检测")
    # edge_image = cv2.Canny(gb_image, 250, 255)
    x_image = cv2.Sobel(gb_image,cv2.CV_16S,1,0)
    y_image = cv2.Sobel(gb_image,cv2.CV_16S,0,1)
    x_image = cv2.convertScaleAbs(x_image)
    y_image = cv2.convertScaleAbs(y_image)
    edge_image = cv2.addWeighted(x_image, 1, y_image, 0.1, 0)
    # cv2.imshow("edge_image", edge_image)
    print("形态学处理：开操作")
    # 先进行腐蚀再进行膨胀
    op_ekernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    op_dkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated_image = cv2.dilate(edge_image, op_dkernel)
    # cv2.imshow("dilated_image", dilated_image)
    eroded_image = cv2.erode(dilated_image, op_ekernel)
    # cv2.imshow("eroded_image", eroded_image)
    print("二值化")
    ret, bw_image = cv2.threshold(eroded_image, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow("bw_image", bw_image)
    print("连通域判断")

    labels = measure.label(bw_image,connectivity=2)#使用8邻域
    small_dst = morphology.remove_small_objects(labels, min_size=2000,connectivity=2)
    rectangle = measure.regionprops(small_dst)
    for region in rectangle:
        minr, minc, maxr, maxc = region.bbox
        for x in range(small_dst.shape[0]):
            for y in range(small_dst.shape[1]):
                if (x>=minr and x<=maxr) and (y>=minc and y<=maxc):
                    small_dst[x,y] = 255

    dst = color.label2rgb(small_dst)
    # print('连通域数目: ', labels.max() + 1)  # 显示连通区域块数(从0开始标记)
    # cv2.imshow("dst",dst)
    # image_info(dst,printflag=True)
    # Otsu 滤波
    dst = img_as_ubyte(dst)
    gray_dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    loc_ret, loc_dst= cv2.threshold(gray_dst, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow("bw_dst",loc_dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return loc_dst



# def find_fa():

# def two_pass(image):

# def domain_test(bw_image):



if __name__=="__main__":
    lp_image = readimage('licenseplate1.jpg',imageflag=cv2.IMREAD_COLOR,showflage=False)
    gray_lp_image = cv2.cvtColor(lp_image, cv2.COLOR_BGR2GRAY)
    # image_info = image_info(lp_image,printflag= False)#获取图像的信息
    loc_image = imagelocation(lp_image)
    license_img = gray_lp_image*loc_image
    cv2.imshow("license",license_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
