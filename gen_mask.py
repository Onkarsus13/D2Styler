import numpy as np
import cv2
from matplotlib import pyplot as plt
from math import ceil

# Given JSON data
data_polygon = {
    "boxes": [
        {
            "type": "polygon",
            "label": "scene-text",
            "x": "134.4846",
            "y": "127.7810",
            "width": "182.7528",
            "height": "46.5097",
            "points": [
                [
                    60.897967894369,
                    107.9840255446543
                ],
                [
                    208.95419041848487,
                    104.52614318596147
                ],
                [
                    225.86102799354043,
                    136.4244948451938
                ],
                [
                    68.82902572983897,
                    151.03588710979798
                ],
                [
                    43.10826673837062,
                    137.3318824144962
                ]
            ]
        }
    ],
    "height": 300,
    "key": "DollarGlen.png",
    "width": 400
}

# Create an empty image with the same size as the original image
mask_polygon = np.zeros((data_polygon["height"], data_polygon["width"]), dtype=np.uint8)

# Process each polygon to create a binary mask
for box in data_polygon["boxes"]:
    if box["type"] == "polygon":
        # Convert points to a format suitable for cv2.fillPoly
        pts = np.array(box["points"], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(mask_polygon, [pts], color=255)  # Fill the polygon area with white

# Display the binary mask for the polygon
plt.imsave("./mask2.png", mask_polygon, cmap='gray')