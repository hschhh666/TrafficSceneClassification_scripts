import numpy as np
import cv2

def pos_converter(height, width, max_time, cur_time, value, height_reserve_percent = 0.05):
    # height 图像高度
    # witdh 图像宽度
    # max_time 显示未来的最长时间
    # cur_time 当前是未来的什么时候
    # value 范围为0-2，表示当前的risk
    # height_reserve_percent 上下留白的比例
    max_value = 2
    x = (width/2) + (cur_time/max_time) * (width/2)
    x = int(x)
    height_reserve = height * height_reserve_percent
    y = height - height_reserve - ((value/max_value) * (height - 2*height_reserve))
    return int(x)-1,int(y)


# 标签转换函数
def label_conventer(l):
    if l == 0: return -1
    if l == 1: return 1
    if l == 2: return 0
    if l == 3: return 2
    if l == 4: return 3
    if l == 5: return 4

# 将 hh:mm:ss 转换成秒
def hms2s(s): 
    h,m,s = s.split(':')
    return int(h)*3600 + int(m) * 60 + int(s)