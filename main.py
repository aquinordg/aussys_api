# 1. Library imports
import os
import shutil
import uvicorn
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import FileResponse

import zipfile

import numpy as np
import pandas as pd

from glob import glob

from tools import split_folders, predict_model

from pydantic import BaseModel

# Setting model and benchmarks
MODELS = glob('models/*')
BENCHMARKS = glob('benchmarks/*')

def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file), 
                       os.path.relpath(os.path.join(root, file), 
                                       os.path.join(path, '..')))

#Create the app object
app = FastAPI()

class status(BaseModel):
    MODELS: list
    BENCHMARKS: list

class message_model(BaseModel):
    msg: str

class message_benchmarks(BaseModel):
    msg: str

class message_predictions(BaseModel):
    msg: str

#Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'AuSSys: Autonomous Search System'}

@app.get('/state', response_model=status)
def state():

    MODELS = glob('models/*')
    BENCHMARKS = glob('benchmarks/*')

    mod = []
    for m in MODELS:
        mod.append(m.split('/')[-1])

    bench = []
    for b in BENCHMARKS:
        bench.append(b.split('/')[-1])
        
    return {"MODELS": mod, "BENCHMARKS": bench}

# Get the benchmarks and models
if os.path.isdir('benchmarks'): sceneries = os.listdir('benchmarks')
else: sceneries = []
if os.path.isdir('models'): models = os.listdir('models')
else: models    = []

@app.post("/upload_model", response_model=message_model)
def upload_model(file: UploadFile = File(...)):
    if not os.path.isdir('models'): 
        os.mkdir('models')

    contents = file.file.read()
    with open(file.filename, 'wb') as f:
        f.write(contents)

    with zipfile.ZipFile(file.filename, 'r') as zip_ref:
        zip_ref.extractall('models')

    os.remove(file.filename)

    return {"msg": f"Model {file.filename} successfully uploaded and extracted."}


@app.post("/upload_benchmarks", response_model=message_benchmarks)
def upload_benchmarks(file: UploadFile = File(...)):
    if not os.path.isdir('files'): 
        os.mkdir('files')

    contents = file.file.read()
    with open(file.filename, 'wb') as f:
        f.write(contents)

    with zipfile.ZipFile(file.filename, 'r') as zip_ref:
        zip_ref.extractall('files')

    os.remove(file.filename)

    if not os.path.isdir('benchmarks'): 
        os.mkdir('benchmarks')

    files = glob('files/*')
    list_files = os.listdir('files/')
    list_benchmarks = os.listdir('benchmarks/')

    for fd, path in zip(files, list_files):
        if path not in list_benchmarks:
            if not os.path.isdir('benchmarks/'+path): 
                os.mkdir('benchmarks/'+path)
            path_split = 'benchmarks/'+path
            split_folders(fd, path_split)
    
    shutil.rmtree('files')

    return {"msg": "Benchmark(s) upload finished."}


@app.post('/run_predictions', response_model=message_predictions)
def run_predictions(scenery: str = Query(enum=sceneries), model: str = Query(enum = models)):

    if not os.path.isdir('results'): 
        os.mkdir('results')

    SCENERY = 'benchmarks/'+scenery
    MODEL   = 'models/'+model
    list_results = os.listdir('results')
    if not os.path.isdir('results'): 
            os.mkdir('results')
        
    if f'results_{model}&{scenery}.csv' not in list_results:
        predict_model(MODEL, SCENERY)

    return {"msg": "Prediction finished."}

@app.post('/download_results')
def download_results():
    with zipfile.ZipFile('results.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipdir('results/', zipf)

    return FileResponse("results.zip")
    
#Run the API with uvicorn
#Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload


