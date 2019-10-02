"""

VFN NTB Image Scoring:
python yolo_video.py --image --model_path outputs/trained_weights_final.h5 --classes_path vott-json-export/classes.txt 
python yolo_video.py --image --model_path outputs/trained_weights_final.h5 --classes_path vott-json-export-20190618/classes.txt 

--img_path ../data-in/vott-json-export/A10%20-%20Ulice.mp4#t=19052.2.jpg

VFN NTB Image Scoring - all annotation_test:
python yolo_video.py --image --model_path logs/001/trained_weights_final.h5 --classes_path vott-json-export/classes.txt --imgs_annotation_path vott-json-export/annotations_test.txt
python yolo_video.py --image --imgs_annotation_path vott-json-export/annotations_test.txt

python yolo_video.py --image --model_path outputs/trained_weights_final.h5 --classes_path vott-json-export/classes.txt --imgs_annotation_path vott-json-export/annotations_test.txt


python yolo_video.py --image --model_path outputs/trained_weights_final.h5 --classes_path vott-json-export-20190618/classes.txt --imgs_annotation_path vott-json-export-20190618/annotations_test.txt



VFN NTB Video scoring:
python yolo_video.py --input ../data-in/video_test/test_110sec.mp4 --output ../data-out/test_110sec_DETECTED_20190808.mp4 --model_path outputs/trained_weights_final.h5 --classes_path vott-json-export/classes.txt
python yolo_video.py --input ../data-in/video_test/test_128sec.mp4 --output ../data-out/test_128sec_DETECTED.mp4 --model_path outputs/trained_weights_final.h5 --classes_path vott-json-export/classes.txt
python yolo_video.py --input ../data-in/video_test/test_24sec.mp4 --output ../data-out/test_24sec.mp4_DETECTED.mp4 --model_path outputs/trained_weights_final.h5 --classes_path vott-json-export/classes.txt
python yolo_video.py --input ../data-in/video_test/test_40sec.mp4 --output ../data-out/test_40sec.mp4_DETECTED.mp4 --model_path outputs/trained_weights_final.h5 --classes_path vott-json-export/classes.txt
python yolo_video.py --input ../data-in/video_test/test_59sec.mp4 --output ../data-out/test_59sec.mp4_DETECTED.mp4 --model_path outputs/trained_weights_final.h5 --classes_path vott-json-export/classes.txt
python yolo_video.py --input ../data-in/video_test/test_66sec.mp4 --output ../data-out/test_66sec.mp4_DETECTED.mp4 --model_path outputs/trained_weights_final.h5 --classes_path vott-json-export/classes.txt
python yolo_video.py --input ../data-in/video_test/test_7sec.mp4 --output ../data-out/test_7sec.mp4_DETECTED.mp4 --model_path outputs/trained_weights_final.h5 --classes_path vott-json-export/classes.txt
python yolo_video.py --input ../data-in/video_test/test_84sec.mp4 --output ../data-out/test_84sec.mp4_DETECTED.mp4 --model_path outputs/trained_weights_final.h5 --classes_path vott-json-export/classes.txt
python yolo_video.py --input ../data-in/video_test/test_ups.mp4 --output ../data-out/test_ups.mp4_DETECTED3.mp4 --model_path outputs/trained_weights_final.h5 --classes_path vott-json-export-20190618/classes.txt

nohup ./score_videos.sh &

YOLO ORIGINAL:
python yolo_video.py --input ../data-in/video_test/test_7sec.mp4 --output ../data-out/test_7sec.mp4_DETECTED_YOLO.mp4
python yolo_video.py --input ../data-in/video_test/test_ups.mp4 --output ../data-out/test_ups.mp4_DETECTED_YOLO.mp4

nohup python yolo_video.py --image --imgs_annotation_path vott-json-export/annotations.txt > out/yolo_labeling.txt &
nohup python yolo_video.py --image --imgs_annotation_path vott-json-export-20190618/annotations.txt > out/yolo_labeling-20190618.txt &
cat out/yolo_labeling.txt | grep "###EXPORT###" > out/yolo_labeling_filtered.txt 
rm out/yolo_labeling* 
"""
import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import os

def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except Exception as e:
            print('EXCEPTION:', str(e))
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            # r_image.show()
            r_image.save("detection.png","PNG")
    yolo.close_session()

# TODO
def detect_img_test_annotation(yolo):
    annotation_path = yolo.imgs_annotation_path
    print(f"yolo.imgs_annotation_path:{yolo.imgs_annotation_path}")
    try:
        # annotation_path = "vott-json-export/annotations_test.txt"
        with open(annotation_path) as f:
            lines = f.readlines()
        for annotation_line in lines:
            line = annotation_line.split()
            fname = line[0].split(" ")[0].split("/")[-1]
            image = Image.open(line[0])
            iw, ih = image.size
            # h, w = input_shape
            box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
            # print("box:",np.array_repr(box).replace('\n', ''))
            r_image = yolo.detect_image(image, image_name=line[0], lbl_box=np.array_repr(box).replace('\n', ''))
            # r_image.show()
            r_image.save("detection.png","PNG")

            draw = ImageDraw.Draw(r_image)

            left, top, right, bottom = box[0][0:-1]
            # x_min,y_min,x_max,y_max
            draw.rectangle(
                [left, top, right, bottom],
                outline="#00ff00")

            del draw

            r_image.save(os.path.join("out",fname+"_detections.png"),"PNG")

        # image = Image.open(img)
    except Exception as e:
            print('EXCEPTION:', str(e))
            print('Open Error! Try again!')
            quit()
       

    yolo.close_session()

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model_path', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes_path', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    # parser.add_argument(
    #     "--img_path", nargs='?', type=str, default="",
    #     help = "[Optional] path to single file for detection"
    # )
    parser.add_argument(
        "--imgs_annotation_path", nargs='?', type=str, default="",
        help = "[Optional] path to annotation file contating images to detect"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        if FLAGS.imgs_annotation_path:
            detect_img_test_annotation(YOLO(**vars(FLAGS)))
        else:
            detect_img(YOLO(**vars(FLAGS)))
        
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
