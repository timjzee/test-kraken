from PIL import Image, ImageDraw, ImageFont
from kraken import blla, rpred
from kraken.lib import models

# try baseline segmentation
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

# refresh image, load model and predict text
im = Image.open('./imgs/Leggerartikel_DVT00_7040_REEKS3.jpg')
model = models.load_any('./model_20.mlmodel')
pred_it = rpred.rpred(model, im, baseline_seg)

new_im = Image.new("RGB", (im.width, im.height), "white")
draw = ImageDraw.Draw(new_im)
font = ImageFont.truetype("/Library/Fonts/Arial Unicode.ttf", 30)

for record in pred_it:
    print(record)
    for pn, p in enumerate(record.prediction, 0):
        draw.text(tuple(record.cuts[pn][0]), p, font=font, fill="black")

new_im.show()
