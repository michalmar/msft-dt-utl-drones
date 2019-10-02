import json
import sys
import argparse
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
        model_path = Model.get_model_path('trained_weights_final_cls4.h5')
    # model_path = "/var/azureml-app/azureml-models/trained_weights_final.h5/1/trained_weights_final.h5"
    
    # init YOLO Class
    yolo = YOLO(model_path=model_path,anchors_path='vott-json-export/yolo_anchors.txt',classes_path='vott-json-export/classes.txt',model_image_size=(416, 416))
    
    # model = yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
    # model.load_weights(model_path) # make sure model, anchors and classes match
    print(f"loaded model from path {model_path} and initialized YOLO class...")

# note you can pass in multiple rows for scoring
def run(raw_data):
    try:
        # data = json.loads(raw_data)['data']
        # data = numpy.array(data)
        # image = Image.open(BytesIO(base64.b64decode(json.loads(raw_data)['data'])))
        image = Image.open(BytesIO(base64.b64decode(raw_data)))
        # image.save("out.png","PNG")
        # image = Image.open("../data-in/vott-json-export/A10%20-%20Namesti.mp4#t=19060.4.jpg")
        out_dict = yolo.detect_image(image, output_detections=True)
        # you can return any datatype as long as it is JSON-serializable
        return out_dict["detections"]
    except Exception as e:
        error = str(e)
        return error

if __name__ == '__main__':
    init(locally=True)


    f ="../data-in/vott-json-export/A10%20-%20Namesti.mp4#t=19060.4.jpg"
    img = Image.open(f)
    # data = base64.b64encode(image_to_byte_array(img)).decode('utf-8')


    buffered = BytesIO()
    img.save(buffered, format="PNG")
    data = base64.b64encode(buffered.getvalue())

    # with open("bas64_test.txt", "wb") as text_file:
    #     text_file.write(data)

    run(data)


# ### TEST LOCALLY ###
# model_path = 'outputs/trained_weights_final.h5'
# # deserialize the model file back into a sklearn model
# num_anchors = 9
# num_classes = 2

# # load model
# model = yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
# model.load_weights(model_path) # make sure model, anchors and classes match
# print("loaded")

# # init YOLO Class
# yolo = YOLO(model_path='outputs/trained_weights_final.h5',anchors_path='model_data/yolo_anchors.txt',classes_path='vott-json-export/classes.txt',model_image_size=(416, 416))

# # run detection
# image = Image.open("../data-in/vott-json-export/A10%20-%20Namesti.mp4#t=19060.4.jpg")
# out_dict = yolo.detect_image(image, output_detections=True)
