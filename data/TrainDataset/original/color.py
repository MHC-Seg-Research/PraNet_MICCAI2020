from PIL import Image
import os
directory = r'/Users/apple/Desktop/data_road/training/gt_image_2'
for pic in os.listdir(directory):
    im = Image.open(directory+"/"+pic)
    imgRgb = im.convert('RGB')
    pixelMap = imgRgb.load()
    img = Image.new( imgRgb.mode, imgRgb.size)
    pixelsNew = img.load()
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            if pixelMap[i,j][2] > 0:
                pixelsNew[i,j] = (255,255,255)
            else:
                pixelsNew[i,j] = (0,0,0)
    img.save('/Users/apple/Desktop/data_road/training/gt_image_2_mod/'+pic)
