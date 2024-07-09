import numpy as np
import cv2
from PIL import Image, ImageDraw
import math
import glob

def poly_to_mask(poly):
    filee = open(poly, 'r')
    mask = np.zeros((512, 512))
    lines = filee.readlines()
    for line in lines:
        line = line.replace('\n', '')
        line = line.split(',')
        line = [int(i) for i in line]

        polygon = line
        width = 512
        height = 512

        img = Image.fromarray(np.zeros((512, 512), dtype='uint8'))
        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
        mask += np.array(img)
    mask = np.expand_dims((mask > 0).astype('uint8'), axis=2)

    mask = mask

    cv2.imwrite('test.png', (np.concatenate((mask, mask, mask), axis=2)*255).astype('uint8'))
    return Image.fromarray(np.concatenate((mask, mask, mask), axis=2)*255)



paths = glob.glob('/DATA/ocr_team_2/onkar2/test/all_text/*.txt')

for i in paths:

    image = poly_to_mask(i)
    image.save('/DATA/ocr_team_2/onkar2/test/all_mask/'+ i.split('/')[-1].split('.')[0] +'.png')

