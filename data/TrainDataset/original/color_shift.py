from PIL import Image
for pic in ('um_road_000000','um_road_000001','um_road_000002'):
    im = Image.open(pic+'.png')
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
    img.save(pic+'_mod.png')


# from PIL import Image
# import numpy as np
# img = Image.open("um_road_000000.png") # open colour image
#
#
# imgRgb = img.convert('RGB')
# pixels = list(imgRgb.getdata())
# width, height = imgRgb.size
# pixels = np.asarray([pixels[i * width:(i + 1) * width] for i in range(height)], dtype=int)
# for i, j in
