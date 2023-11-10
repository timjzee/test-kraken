from PIL import Image, ImageDraw

from kraken import binarization
from kraken import pageseg

# can be any supported image format and mode
im = Image.open('./imgs/Leggerartikel_DVT00_7040_REEKS3.jpg')
bw_im = binarization.nlbin(im)

seg = pageseg.segment(bw_im)

draw = ImageDraw.Draw(im)

line_color = (255, 0, 0)
line_width = 2

for box in seg["boxes"]:
    box_coordinates = [(box[0], box[1]), (box[2], box[3])]
    draw.rectangle(box_coordinates, outline="red", width=line_width)

im.show()

# try baseline segmentation

from kraken import blla
from kraken.lib import vgsl
from kraken import serialization

im = Image.open('./imgs/Leggerartikel_DVT00_7040_REEKS3.jpg')
baseline_seg = blla.segment(im)

draw = ImageDraw.Draw(im)

for l in baseline_seg["lines"]:
    bl = l["baseline"]
    for i in range(len(bl) - 1):
        draw.line([tuple(bl[i]), tuple(bl[i+1])], fill="blue", width=4)
    bnd = [tuple(b) for b in l["boundary"]]
    draw.polygon(bnd, outline="red", fill=None)


im.show()
        