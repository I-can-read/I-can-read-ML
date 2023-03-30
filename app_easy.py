# -*- coding: utf-8 -*-
import sys
import os

sys.path.append('./deep_text_recognition_benchmark')
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./i-can-read-379204-ce1c5c2f12f5.json"

import uvicorn
import torch
import pickle
import configparser
import requests
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile
from preprocessImage import crop_image, preprocess_image

app = FastAPI(max_request_size=1024*1024*1024)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = './saved_models/pretrained/best_accuracy.pth'

dir_name = "words"
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

config = configparser.ConfigParser()
config.read('config.ini')

host = config.get('server', 'host')
port = config.getint('server', 'port')


@app.get("/")
async def root():
    return {"message": f"Server running on {host}:{port}"}


@app.post('/api/v1/menu/extract')
async def my_async_function(file: UploadFile):
    menu = preprocess_image(file)
    text_list = crop_image(menu)
    print(text_list)
    return text_list


if __name__ == '__main__':
    uvicorn.run(app, host=host, port=port)