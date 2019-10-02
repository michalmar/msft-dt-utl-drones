"""
Retrain the YOLO model for your own dataset.

python train.py
"""

import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data

from azureml.core.run import Run
import os

import keras.callbacks as KC
class LossAndErrorPrintingCallback(KC.Callback):

  def on_epoch_end(self, epoch, logs=None):
    run = Run.get_context()
    run.log("EPOCH_loss", logs["loss"])   
    run.log("EPOCH_val_loss", logs["val_loss"])
    # print('MMA CALLBACK: The average loss for epoch {} is {:7.2f}.'.format(epoch, logs['loss']))

def _main(data_folder, model_folder, annotation_path, log_dir, classes_path, anchors_path, epochs_frozen=50, epochs_unfrozen=150):
    
    run = Run.get_context()
    os.makedirs('outputs', exist_ok=True)

    annotation_path = annotation_path
    log_dir = log_dir
    classes_path = classes_path
    anchors_path = anchors_path
        
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    run.log('data_folder', data_folder)
    run.log('model_folder', model_folder)

    run.log('epochs_frozen', epochs_frozen)
    run.log('epochs_unfrozen', epochs_unfrozen)

    run.log('annotation_path', annotation_path)
    run.log('log_dir', log_dir)
    run.log('classes_path', classes_path)
    run.log('anchors_path', anchors_path)
    run.log('num_classes', num_classes)
    

    input_shape = (416,416) # multiple of 32, hw
    run.log('input_shape', input_shape)
    
    
    is_tiny_version = len(anchors)==6 # default setting
    run.log('is_tiny_version', is_tiny_version)

    
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path=os.path.join(model_folder,"tiny_yolo_weights.h5"))
    else:
        model = create_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path=os.path.join(model_folder,"yolo.h5")) # make sure you know what you freeze

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    custom_log = LossAndErrorPrintingCallback()

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    run.log('num_val', num_val)
    run.log('num_train', num_train)
    
    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = 32
        # epochs=50
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes, data_folder),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes, data_folder),
                validation_steps=max(1, num_val//batch_size),
                epochs=epochs_frozen,
                initial_epoch=0,
                callbacks=[logging, checkpoint, custom_log])
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')
        # save model in the outputs folder so it automatically get uploaded
        # model.save_weights(os.path.join('./outputs/' , 'trained_weights_stage_1.h5'))
        # with open(model_file_name, "wb") as file:
        #     joblib.dump(value=reg, filename=os.path.join('./outputs/',
        #                                              model_file_name))

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        # epochs=150
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
        print('Unfreeze all of the layers.')

        batch_size = 16 # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes, data_folder),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes, data_folder),
            validation_steps=max(1, num_val//batch_size),
            epochs=epochs_unfrozen,
            initial_epoch=epochs_frozen,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping, custom_log])
        model.save_weights(log_dir + 'trained_weights_final.h5')

    # Further training if needed.


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers)-2)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, data_folder):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(os.path.join(data_folder, annotation_lines[i]), input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes, data_folder):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, data_folder)

# if __name__ == '__main__':
#     _main()



# class YOLO defines the default value, so suppress any default here
parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
'''
Command line options
'''
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
parser.add_argument('--model-folder', type=str, dest='model_folder', help='data folder mounting point')

parser.add_argument('--annotation_path', type=str, dest='annotation_path', help='path to training files annotation')
parser.add_argument('--log_dir', type=str, dest='log_dir', help='where logs and intermediate model are placed')
parser.add_argument('--classes_path', type=str, dest='classes_path', help='path to training classes') 
parser.add_argument('--anchors_path', type=str, dest='anchors_path', help='path to training anchors') 
parser.add_argument('--epochs_frozen', type=str, dest='epochs_frozen', help='epochs on frozen heads') 
parser.add_argument('--epochs_unfrozen', type=str, dest='epochs_unfrozen', help='epochs on unfrozen heads - all net') 

# parser.add_argument('--regularization', type=float, dest='reg', default=0.01, help='regularization rate')
args = parser.parse_args()

data_folder = args.data_folder
model_folder = args.model_folder
annotation_path = args.annotation_path
log_dir = args.log_dir
classes_path = args.classes_path
anchors_path = args.anchors_path
epochs_frozen = int(args.epochs_frozen)
epochs_unfrozen = int(args.epochs_unfrozen)

from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

print(f'get_available_gpus:{get_available_gpus()}')

_main(data_folder=data_folder, model_folder=model_folder, annotation_path = annotation_path, log_dir = log_dir, classes_path = classes_path, anchors_path = anchors_path, epochs_frozen=epochs_frozen, epochs_unfrozen=epochs_unfrozen)