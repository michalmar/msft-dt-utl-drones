# DOTS - Utils - Drones V2

## Workflow:
### 01 data prep (convert VOTT->YOLO annotation)
1. get the exported`*.JSON` file from VOTT tagging tool and copy somwhere on FS (ideally new folder...)
1. run the conversion Python script `python convert_vott2yolo_annotations.py`
1. change of annotation files in  `aml_prj` (removed path) 
    `../data-in/vott-json-export-20190808/A10%20-%20Namesti%202.mp4#t=16240.6.jpg 459,74,704,249,0`
    will be
    `A10%20-%20Namesti%202.mp4#t=16240.6.jpg 459,74,704,249,0`
1. copy images to blob (data-in, remeber the folder structure)

### 02 run the TRAINING
1. modify `train_aml_wrapper.py` (TODO: create paratmer for that): line 98 -> put correct name of the folder
1. run remote training
`python train_aml_wrapper.py --aml_compute amlgpu-low --annotation_path 'vott-json-export-20190808/annotations.txt' --log_dir 'outputs/' --classes_path 'vott-json-export-20190808/classes.txt' --anchors_path 'vott-json-export-20190808/yolo_anchors.txt' --epochs_frozen 133 --epochs_unfrozen 666`
> note: don't forget to use proper name of folders (in paths)
> note: set parameters  epochs_frozen / epochs_unfrozen accordingly or leave as default
```
        usage: train_aml_wrapper.py [-h] [--aml_compute AML_COMPUTE]
                                    [--annotation_path ANNOTATION_PATH]
                                    [--log_dir LOG_DIR] [--classes_path CLASSES_PATH]
                                    [--anchors_path ANCHORS_PATH]
                                    [--epochs_frozen EPOCHS_FROZEN]
                                    [--epochs_unfrozen EPOCHS_UNFROZEN]

        optional arguments:
        -h, --help            show this help message and exit
        --aml_compute AML_COMPUTE
                                Set specific AML Compute, using default if not set.
        --annotation_path ANNOTATION_PATH
                                path to training files annotation
        --log_dir LOG_DIR     where logs and intermediate model are placed
        --classes_path CLASSES_PATH
                                path to training classes
        --anchors_path ANCHORS_PATH
                                path to training anchors
        --epochs_frozen EPOCHS_FROZEN
                                epochs on frozen heads
        --epochs_unfrozen EPOCHS_UNFROZEN
                                epochs on unfrozen heads - all net
```

### 03 Inference

**single image**
`python yolo_video.py --image --model_path outputs/trained_weights_final.h5 --classes_path vott-json-export-20190618/classes.txt `

**Video**
`python yolo_video.py --input ../data-in/video_test/test_110sec.mp4 --output ../data-out/test_110sec_DETECTED_20190808.mp4 --model_path outputs/trained_weights_final.h5 --classes_path vott-json-export/classes.txt`
> note: number of classes must be same as for training (otherwise error with mismatch dimension can appear)
> better to use correct classess file from training `--classes_path ./vott-json-export-20190808/classes.txt`



# --------------------------------------------------------------------------------

# YOLO v3 Keras ()
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

## Introduction

A Keras implementation of YOLOv3 (Tensorflow backend) inspired by [allanzelener/YAD2K](https://github.com/allanzelener/YAD2K).


---

## Quick Start

1. Download YOLOv3 weights from [YOLO website](http://pjreddie.com/darknet/yolo/).
2. Convert the Darknet YOLO model to a Keras model.
3. Run YOLO detection.

```
wget https://pjreddie.com/media/files/yolov3.weights
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
python yolo_video.py [OPTIONS...] --image, for image detection mode, OR
python yolo_video.py [video_path] [output_path (optional)]
```

For Tiny YOLOv3, just do in a similar way, just specify model path and anchor path with `--model model_file` and `--anchors anchor_file`.

### Usage
Use --help to see usage of yolo_video.py:
```
usage: yolo_video.py [-h] [--model MODEL] [--anchors ANCHORS]
                     [--classes CLASSES] [--gpu_num GPU_NUM] [--image]
                     [--input] [--output]

positional arguments:
  --input        Video input path
  --output       Video output path

optional arguments:
  -h, --help         show this help message and exit
  --model MODEL      path to model weight file, default model_data/yolo.h5
  --anchors ANCHORS  path to anchor definitions, default
                     model_data/yolo_anchors.txt
  --classes CLASSES  path to class definitions, default
                     model_data/coco_classes.txt
  --gpu_num GPU_NUM  Number of GPU to use, default 1
  --image            Image detection mode, will ignore all positional arguments
```
---

4. MultiGPU usage: use `--gpu_num N` to use N GPUs. It is passed to the [Keras multi_gpu_model()](https://keras.io/utils/#multi_gpu_model).

## Training

1. Generate your own annotation file and class names file.  
    One row for one image;  
    Row format: `image_file_path box1 box2 ... boxN`;  
    Box format: `x_min,y_min,x_max,y_max,class_id` (no space).  
    For VOC dataset, try `python voc_annotation.py`  
    Here is an example:
    ```
    path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
    path/to/img2.jpg 120,300,250,600,2
    ...
    ```

2. Make sure you have run `python convert.py -w yolov3.cfg yolov3.weights model_data/yolo_weights.h5`  
    The file model_data/yolo_weights.h5 is used to load pretrained weights.

3. Modify train.py and start training.  
    `python train.py`  
    Use your trained weights or checkpoint weights with command line option `--model model_file` when using yolo_video.py
    Remember to modify class path or anchor path, with `--classes class_file` and `--anchors anchor_file`.

If you want to use original pretrained weights for YOLOv3:  
    1. `wget https://pjreddie.com/media/files/darknet53.conv.74`  
    2. rename it as darknet53.weights  
    3. `python convert.py -w darknet53.cfg darknet53.weights model_data/darknet53_weights.h5`  
    4. use model_data/darknet53_weights.h5 in train.py

---

## Some issues to know

1. The test environment is
    - Python 3.5.2
    - Keras 2.1.5
    - tensorflow 1.6.0

2. Default anchors are used. If you use your own anchors, probably some changes are needed.

3. The inference result is not totally the same as Darknet but the difference is small.

4. The speed is slower than Darknet. Replacing PIL with opencv may help a little.

5. Always load pretrained weights and freeze layers in the first stage of training. Or try Darknet training. It's OK if there is a mismatch warning.

6. The training strategy is for reference only. Adjust it according to your dataset and your goal. And add further strategy if needed.

7. For speeding up the training process with frozen layers train_bottleneck.py can be used. It will compute the bottleneck features of the frozen model first and then only trains the last layers. This makes training on CPU possible in a reasonable time. See [this](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) for more information on bottleneck features.
