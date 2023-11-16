import os
import json
import zipfile
import shutil
import numpy as np
import pandas as pd
import splitfolders

from glob import glob
from keras import backend
from tensorflow import expand_dims
from tensorflow.keras import preprocessing, models

import warnings
warnings.filterwarnings("ignore")

def recall(y_true, y_pred):
    true_positives = backend.sum(
        backend.round(backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = backend.sum(backend.round(backend.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + backend.epsilon())


def precision(y_true, y_pred):
    true_positives = backend.sum(
        backend.round(backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = backend.sum(
        backend.round(backend.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + backend.epsilon())


def fbeta(y_true, y_pred, beta=1.0):
    pre = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)

    score = (1 + beta*2) * (pre * rec) / ((beta*2 * pre) + rec)
    return score

def predict(model, dataset, image_size):
    nil = glob(f'{dataset}/test/NIL/*.jpg')
    pod = glob(f'{dataset}/test/POD/*.jpg')

    X_test = nil + pod
    y_test = [ 0 for _ in range(len(nil)) ] + [ 1 for _ in range(len(pod)) ]
    y_pred = []
    y_pred_proba = []

    # predict dataset test
    id = []
    for i in range(len(X_test)):
        path_image = X_test[i]
        id.append(path_image.split('/')[-1])

        if len(image_size) == 2: image = preprocessing.image.load_img(path_image, target_size=image_size, color_mode="grayscale")
        else: image = preprocessing.image.load_img(path_image, target_size=image_size)
        image = expand_dims(image, 0)

        prediction = model.predict(image, verbose=0)
        y_pred_proba.append(prediction[:, 1][0])
        y_pred.append(prediction.argmax(axis=1)[0])

    return id, y_test, y_pred, y_pred_proba

def predict_model(model: str, dataset: str):
    model_path = model.split('.zip')[0]
    with zipfile.ZipFile(model, 'r') as zip_ref:
        zip_ref.extractall(model_path)
    f = open(model_path+'/metadata.json')
    data = json.load(f)
    f.close()

    model_name   = model_path.split('/')[-1]
    dataset_name = dataset.split('/')[-1]

    model = models.load_model(model_path+'/model', custom_objects={'fbeta': fbeta})
    
    results = pd.DataFrame()
    for i in range(10):
        res_aux = pd.DataFrame()
        path = f'{dataset}/split_{i+1:02}'
        id, y_test, y_pred, y_pred_proba = predict(model, path, image_size = eval(data['model_input']))

        res_aux = pd.DataFrame({'id': id,
                                'model': model_name,
                                'dataset': dataset_name,
                                'fold': i+1,
                                'expected': y_test,
                                'predicted': y_pred_proba})
        
        results = pd.concat([results, res_aux], ignore_index=True)

    results.to_csv(f'results/results_{model_name}&{dataset_name}.csv', index = False)
    shutil.rmtree(model_path)

def split_folders(path_dataset, path_split):
    list_seed = []
    for i in range(10):
        while True:  
            seed = np.random.randint(32, 230)
            if seed not in list_seed:
                list_seed.append(seed)
                break
            
            folder = f'{path_split}/split_{i+1:02}'
            os.system(f'mkdir -p {path_split}')
            val_size = min(len(os.listdir(path_dataset+'/test/POD')), len(os.listdir(path_dataset+'/test/NIL')))//10
            path_test = path_dataset+'/test'
            splitfolders.fixed(path_test, folder, seed=seed, fixed=(val_size, val_size))
