# Test for eval models for MSc lab (CV, AI, ML)
# Created by Ivan Martsilenko MMAI-1

from time import time
from tensorflow.keras.models import load_model
from tensorflow import distribute
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import optimizers
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.utils import image_dataset_from_directory
from shutil import copyfile
import os
from progress.bar import IncrementalBar
from tensorflow.keras.preprocessing import image
import numpy as np

from datetime import datetime

#model_path = 'e:\\Uni\\NLP\\1st\\models\\resnet_50_45epochs.h5'
#test_dir = 'e:\\Uni\\Demo\\landmark_data_demo'
#output_dir = os.path.join('e:\\uni\\Demo', 'output')

def _create_ResNet_model(model_path):
    strategy = distribute.MirroredStrategy(devices=['/gpu:0'])
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
        model = ResNet50(classes= 400,
                        weights = None,
                        input_shape=(224, 224, 3))

        model.trainable = True
        model.compile(optimizer=optimizers.RMSprop(learning_rate=2e-5),
                loss='categorical_crossentropy',
                metrics=['accuracy', Precision(), Recall()])
    
    model = load_model(model_path)
    return model

def _create_MobNetV2_model(model_path):
    strategy = distribute.MirroredStrategy(devices=['/gpu:0'])
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
        MobileNetV2Model = MobileNetV2(include_top=False,
                            weights="imagenet",
                            input_shape=(224, 224, 3)
        )

        MobileNetV2Model.trainable = True
        model = Sequential()
        model.add(MobileNetV2Model)
        model.add(GlobalAveragePooling2D())
        model.add(Dense(100, activation='softmax'))
        model.compile(optimizer=optimizers.RMSprop(learning_rate=2e-5),
                loss='categorical_crossentropy',
                metrics=['accuracy', Precision(), Recall()])
    
    model = load_model(model_path)
    return model


def _predict_labels(set_dir: str, model) -> None:
    start_time = datetime.now()
    predictions = model.predict(image_dataset_from_directory(set_dir, image_size=(224,224)))
    total_time = datetime.now() - start_time
    print(f"Total images: {len(predictions)}\nTotal time: {total_time}\nAvarage time per image: {total_time/len(predictions)}")

def _create_roots(output_dir: str) -> None:
    try:
        os.mkdir(os.path.join(output_dir))
    except:
        pass

def _sort_files(set_dir:str, output_dir: str,  model) -> None:    # Bug !
    i = 0
    for root, dirs, files in os.walk(set_dir):
        for dir in dirs:
            for dir_root, dir_dirs, dir_files in os.walk(os.path.join(set_dir, dir)):
                for file in dir_files:
                    img = image.load_img(os.path.join(set_dir, dir, file),target_size=(224,224,3))
                    img = image.img_to_array(img)
                    img = img/255
                    label = model.predict(img.reshape(1, 224,224, 3))
                    label = np.argsort(label[0])[-1]
                    try:
                        os.mkdir(os.path.join(output_dir, str(label)))
                    except:
                        pass
                    copyfile(os.path.join(set_dir, dir, file), os.path.join(output_dir, str(label), file))
                    i+=1

def sort_landmark(model_path: str, set_dir: str, output_dir: str) -> None:
    #model = _create_MobNetV2_model(model_path)
    model = _create_ResNet_model(model_path)
    prediction_labels = _predict_labels(set_dir, model)
    _create_roots(output_dir, prediction_labels)
    _sort_files(set_dir, output_dir, prediction_labels)

if __name__ == '__main__':
    #sort_landmark( 'e:\\Uni\\Demo\\lanmark_models\\resnet_50_45epochs.h5','e:\\Uni\\Demo\\landmark_data_demo', os.path.join('e:\\uni\\Demo', 'output'))
    #model = _create_ResNet_model('e:\\Uni\\Demo\\lanmark_models\\resnet_50_45epochs.h5')
    #_create_roots('e:\\uni\\Demo\\output')
    model = _create_MobNetV2_model('e:\\Uni\\Demo\\lanmark_models\\mobile_net_v2_25epochs.h5')
    #_sort_files('e:\\Uni\\Demo\\landmark_data_demo', os.path.join('e:\\uni\\Demo', 'output'), model)

    set_dir = 'E:\\datasets\\working\\validation'
    _predict_labels(set_dir, model)