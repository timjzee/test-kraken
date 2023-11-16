import glob
import os
import json
from PIL import Image
from kraken import blla, rpred
from kraken.lib import models

file_paths = glob.glob("imgs/kadaster_scans/*/*.jpg")

# load model
model = models.load_any('./model_20.mlmodel')

for fpn, fp in enumerate(file_paths, 1):
    print("{}/{}  -  {}".format(fpn, len(file_paths), fp))
    # load img
    im = Image.open(fp)
    # segment text
    baseline_seg = blla.segment(im)
    # predict text
    pred_it = rpred.rpred(model, im, baseline_seg)
    # create json
    data = []
    for record in pred_it:
        chunk = {
            "location": record.baseline,
            "text": record.prediction
        }
        data.append(chunk)
    # create output folders if they don't exist
    ofs = fp.split("/")[1:-1]
    for ofn, of in enumerate(ofs):
        ofp = "output/" + "/".join(ofs[:ofn]) + "/" + of
        if not os.path.exists(ofp):
            os.makedirs(ofp)
    # write json
    ofname = fp.split("/")[-1].split(".")[0] + ".json"
    with open(ofp + "/" + ofname, 'w') as json_file:
        json.dump(data, json_file, indent=2)