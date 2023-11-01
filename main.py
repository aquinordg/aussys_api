# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:40:41 2020

@author: win10
"""

# 1. Library imports
import os
#import shutil
import uvicorn
from fastapi import FastAPI, File, UploadFile, Query
import zipfile


import numpy as np
import pandas as pd

from glob import glob

from tools import split_folders, predict_model

# Variables
thresholds = [ .25, 0.5, 0.75 ]
list_thr   = np.arange(0, 1, 0.005).tolist()

# Setting model and benchmarks
MODELS = glob('models/*')
BENCHMARKS = glob('benchmarks/*')

#Create the app object
app = FastAPI()

#Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'AuSSys: Autonomous Search System'}

@app.post('/state')
def state():
    mod = []
    for m in MODELS:
        mod.append(m.split('/')[-1])

    bench = []
    for b in BENCHMARKS:
        bench.append(b.split('/')[-1])
    return {'message': f'MODELS: {mod} BENCHMARKS: {bench}'}

# Get the benchmarks and models
if os.path.isdir('benchmarks'): sceneries = os.listdir('benchmarks')
else: sceneries = []
if os.path.isdir('models'): models = os.listdir('models')
else: models    = []

@app.post("/upload_model")
def upload_model(file: UploadFile = File(...)):
    try:
        if not os.path.isdir('models'): 
            os.mkdir('models')

        contents = file.file.read()
        with open(file.filename, 'wb') as f:
            f.write(contents)

        with zipfile.ZipFile(file.filename, 'r') as zip_ref:
            zip_ref.extractall('models')

        os.remove(file.filename)

    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    return {"message": f"{file.filename} successfully uploaded and extracted."}


@app.post("/upload_benchmarks")
def upload_benchmarks(file: UploadFile = File(...)):
    try:
        if not os.path.isdir('files'): 
            os.mkdir('files')

        contents = file.file.read()
        with open(file.filename, 'wb') as f:
            f.write(contents)

        with zipfile.ZipFile(file.filename, 'r') as zip_ref:
            zip_ref.extractall('files')

        os.remove(file.filename)

    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

        if not os.path.isdir('benchmarks'): 
            os.mkdir('benchmarks')

        files = glob('files/*')
        list_sets = os.listdir('files/')
        for fd, path in zip(files, list_sets):
            if not os.path.isdir('benchmarks/'+path): 
                os.mkdir('benchmarks/'+path)
            path_split = 'benchmarks/'+path
            split_folders(fd, path_split)

    return {"message": f"{file.filename} successfully uploaded and extracted."}


@app.post('/run_predictions')
def run_predictions(scenery: str = Query(enum=sceneries), model: str = Query(enum = models)):
    SCENERY = 'benchmarks/'+scenery
    MODEL   = 'models/'+model
    if not os.path.isdir('results'): 
            os.mkdir('results')
        
    predict_model(MODEL, SCENERY)

    #shutil.make_archive('results', 'zip', 'results')

    return {"message": f"Results ready"}
    
#Run the API with uvicorn
#Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload


