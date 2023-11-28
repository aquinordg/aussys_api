# 1. Library imports
import os
import shutil
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse

import zipfile

import numpy as np
import pandas as pd

from glob import glob

from tools import split_folders, predict_model

from pydantic import BaseModel

# Setting model and benchmarks
MODELS = glob('models/*')
BENCHMARKS = glob('scenarios/*')


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
    SCENARIOS: list
    RESULTS: list


class message_model(BaseModel):
    msg: str

#Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'AuSSys: Autonomous Search System'}


@app.get('/state', response_model=status)
def state():

    MODELS = glob('models/*')
    SCENARIOS = glob('scenarios/*')
    RESULTS = glob('results/*')

    res = []
    for r in RESULTS:
        res.append(r.split('/')[-1])

    mod = []
    for m in MODELS:
        mod.append(m.split('/')[1].split('.')[0])

    sce = []
    for s in SCENARIOS:
        sce.append(s.split('/')[-1])

    return {"MODELS": mod, "SCENARIOS": sce, "RESULTS": res}


# Get the benchmarks and models
if os.path.isdir('scenarios'): sceneries = os.listdir('scenarios')
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


@app.post("/upload_scenarios", response_model=message_model)
def upload_scenarios(file: UploadFile = File(...)):
    if not os.path.isdir('files'):
        os.mkdir('files')

    contents = file.file.read()
    with open(file.filename, 'wb') as f:
        f.write(contents)

    with zipfile.ZipFile(file.filename, 'r') as zip_ref:
        zip_ref.extractall('files')

    os.remove(file.filename)

    if not os.path.isdir('scenarios'):
        os.mkdir('scenarios')

    files = glob('files/test_set/*')
    list_files = os.listdir('files/test_set/')
    list_scenarios = os.listdir('scenarios/')

    for fd, path in zip(files, list_files):
        if path not in list_scenarios:
            if not os.path.isdir('scenarios/'+path):
                os.mkdir('scenarios/'+path)
            path_split = 'scenarios/'+path
            split_folders(fd, path_split)

    shutil.rmtree('files')

    return {"msg": "Upload finished."}


@app.post('/run_predictions', response_model=message_model)
def run_predictions(scenery: str, model: str):

    if not os.path.isdir('results'):
        os.mkdir('results')

    SCENERY = 'scenarios/' + scenery
    MODEL   = 'models/' + model + '.zip'
    list_results = os.listdir('results')
    if not os.path.isdir('results'):
        os.mkdir('results')

    if f'results_{model}&{scenery}.csv' not in list_results:
        predict_model(MODEL, SCENERY)

    return {"msg": "Prediction finished."}


@app.post('/download_results', response_model=message_model)
def download_results(scenery: str, model: str):

    filename = f'results/results_{model}&{scenery}.csv'

    if os.path.isfile(filename):
        return FileResponse(filename, media_type="text/csv")

    return {"msg": "Result not available."}


#Run the API with uvicorn
#Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

#uvicorn app:app --reload
