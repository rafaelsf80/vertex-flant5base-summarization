from fastapi import FastAPI, Request

import json
import numpy as np
import pickle
import os

from transformers import AutoTokenizer, T5ForConditionalGeneration

app = FastAPI()

@app.get(os.environ['AIP_HEALTH_ROUTE'], status_code=200)
def health():
    return {"status": "healthy"}

@app.post(os.environ['AIP_PREDICT_ROUTE'])
async def predict(request: Request):
    body = await request.json()

    instances = body["instances"]

    # Load tokenizer
    model_id = '../model-output-flan-t5-base'
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load model
    model_name = '../model-output-flan-t5-base'
    model = T5ForConditionalGeneration.from_pretrained(model_name) 

    outputs = []
    for instance in instances:

        generated = model.generate(**tokenizer(instance, return_tensors="pt", padding=True), max_new_tokens=50)
        outputs.append([tokenizer.decode(t, skip_special_tokens=True) for t in generated])   

    return {"predictions": [outputs]}