import os
import numpy as np
import pandas as pd
import splitfolders
from glob import glob

from tensorflow import expand_dims
from tensorflow.keras import preprocessing, models

import warnings
warnings.filterwarnings("ignore")

def predict(model, dataset, image_size: tuple = (64, 64), color_mode: str = 'grayscale'):
    nil = glob(f'{dataset}/test/nil/*.jpg')
    pod = glob(f'{dataset}/test/pod/*.jpg')

    X_test = nil + pod
    y_test = [ 0 for _ in range(len(nil)) ] + [ 1 for _ in range(len(pod)) ]
    y_pred = []
    y_pred_proba = []

    # predict dataset test
    id = []
    for i in range(len(X_test)):
        path_image = X_test[i]
        id.append(path_image.split('/')[-1])

        image = preprocessing.image.load_img(path_image, color_mode=color_mode, target_size=image_size)
        image = preprocessing.image.img_to_array(image) / 255
        image = expand_dims(image, 0)

        prediction = model.predict(image, verbose=0)
        y_pred_proba.append(prediction[:, 1][0])
        y_pred.append(prediction.argmax(axis=1)[0])

    return id, y_test, y_pred, y_pred_proba

def predict_model(model: str, dataset: str, image_size: tuple = (64, 64), color_mode: str = 'grayscale'):
    model_name = model.split('/')[-1]
    dataset_name = dataset.split('/')[-1]
    model = models.load_model(model)
    
    results = pd.DataFrame()
    for i in range(10):
        res_aux = pd.DataFrame()
        path = f'{dataset}/split_{i+1:02}'
        id, y_test, y_pred, y_pred_proba = predict(model, path)

        res_aux = pd.DataFrame({'id': id,
                                'model': model_name,
                                'dataset': dataset_name,
                                'fold': i+1,
                                'expected': y_test,
                                'predicted': y_pred_proba})
        
        results = pd.concat([results, res_aux], ignore_index=True)

    results.to_csv(f'results/results_{model_name}&{dataset_name}.csv', index = False)

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
            val_size = min(len(os.listdir(path_dataset+'/pod')), len(os.listdir(path_dataset+'/pod')))//10
            splitfolders.fixed(path_dataset, folder, seed=seed, fixed=(val_size, val_size))