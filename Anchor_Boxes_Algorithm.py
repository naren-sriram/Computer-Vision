import anchor_boxes
from anchor_boxes import Anchor_boxes
import cv2
import numpy as np
from matplotlib import pyplot as plt
#Create an object of Anchor_Boxes class
AnchorBox = Anchor_boxes(

    img_height = 300,

   img_width = 300,

   this_scale = 0.13,

   next_scale = 1,

   aspect_ratios=[1.0],

   clip_boxes = True,

   this_steps = 16,

   this_offsets = 0.5,

    two_boxes_for_ar1=False,

    variances=None,

    coords='centroids',

    normalize_coords=False,
)

path = r'C:\Users\nasriram\Test_Original_Image.jpg'
#Read Input Image
input_img = cv2.imread(path)

#For reference: print(input_img.shape)

#Control every nth anchor box parameter here
Overlay_every_nth_box = 4

#Call AnchorBox main function and pass the above image as input to it.
anchor_boxes = AnchorBox.call(input_img, overlay=Overlay_every_nth_box)

#plt.figure(figsize=(20,12))
#plt.imshow(input_img)

#Draw the genrated anchor boxes on the input image
for k in range(anchor_boxes.shape[0]):
    for l in range(anchor_boxes.shape[1]):
        cx,cy,w,h = anchor_boxes[k][l][0]
        input_img = cv2.rectangle(input_img, (int(cx-w/2), int(cy-h/2)), (int(cx+w/2), int(cy+h/2)), (0,0,255), 1)

#Save the image in the same path as above.
cv2.imwrite(f'Anchor_Sample{Overlay_every_nth_box}.jpg', input_img)



