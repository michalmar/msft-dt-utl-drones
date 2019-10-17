import json
import sys
import argparse

sys.path.append('./aml_deploy_prj')

from yolo import YOLO, detect_video
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import os

from keras.models import load_model
from keras.layers import Input
from azureml.core.model import Model
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from io import BytesIO
import base64


def init(locally=False):
    global yolo
    # get model from service
    if (locally):
        model_path = 'outputs/trained_weights_final.h5'
    else: 
        model_path = Model.get_model_path('dronesv2.h5')
        
    # init YOLO Class
    yolo = YOLO(model_path=model_path,anchors_path=os.path.join("aml_deploy_prj","model_data","yolo_anchors.txt"),classes_path=os.path.join("aml_deploy_prj","model_data","classes.txt"),model_image_size=(416, 416))
    # print(f"INFO: loaded model from path {model_path} and initialized YOLO class...")

# note you can pass in multiple rows for scoring
def run(raw_data):
    try:
        # print("opening image")
        image = Image.open(BytesIO(base64.b64decode(raw_data)))
        # print("detecting")
        out_dict = yolo.detect_image(image, output_detections=True)
        # print("done")
        # print(out_dict)
        # print(out_dict["detections"])
        # you can return any datatype as long as it is JSON-serializable
        return out_dict["detections"]
    except Exception as e:
        error = str(e)
        print(f"error: {e}")
        return error

if __name__ == '__main__':
    init(locally=True)


    f ="data\WIN_20190920_12_59_28_Pro.jpg"
    img = Image.open(f)
    # data = base64.b64encode(image_to_byte_array(img)).decode('utf-8')


    buffered = BytesIO()
    img.save(buffered, format="PNG")
    data = base64.b64encode(buffered.getvalue())

    # with open("bas64_test.txt", "wb") as text_file:
    #     text_file.write(data)

    run(data)

