#  Finetuning FlanT5Base and online deployment in Vertex AI

This code shows how to **finetune a Flan-T5-Base model** (stored in Hugging Face) for **SAMSum** dataset (summary of conversations in English).
This code uses **Vertex AI Training with 1xA100 GPU** (40 GB HBM) for finetuning, and **Vertex AI prediction** for online predictions.


## Dataset

The **SAMSum dataset** contains about 16k messenger-like conversations with summaries. Conversations were created and written down by linguists **fluent in English**. This use case is similar for example to the conversation summaries feature [available in Google Chat](https://support.google.com/chat/answer/12918975?hl=en).


## Finetuning

Using libraries from Hugging Face, the code sample **fine tunes a Flan-T5-Base model** on the **SAMSum dataset for English conversations**. Dataset is [stored in Hugging Face](https://huggingface.co/datasets/samsum). The Flan T5 Base model is the `google/flan-t5-base` [stored in Hugging face](https://huggingface.co/google/flan-t5-base), and the finetuned version will be moved to Vertex AI Model registry.

Run the finetune process with:
```py
python3 custom_training.py
```

The model is fine tuned on **Vertex AI with 1xV100 NVIDIA GPU** (40 GB HBM), using Vertex AI Training. The code launches a Training pipeline, a type of Vertex AI job, which runs the following three steps: creates a Managed Dataset (not created here), Training, and Model Upload to model Registry (the model is actually uploaded to GCS first, and to Model Registry in the next step): 
```py
job = aiplatform.CustomTrainingJob(
    display_name="flan_t5_base_finetuning_gpu_tensorboard",
    script_path="flant5base_trainer.py",
    requirements=["py7zr==0.20.4",
                  "nltk==3.7",
                  "evaluate==0.4.0",
                  "rouge_score==0.1.2", 
                  "transformers==4.25.1",
                  "tensorboard==2.11.2",
                  "datasets==2.9.0",
                  "google-cloud-storage==2.7.0"],
    container_uri="europe-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-10:latest",
    model_serving_container_image_uri="europe-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.1-10:latest",
)

model = job.run(
    model_display_name='flan-t5-base-finetuning-gpu-tensorboard',
    replica_count=1,
    service_account = SERVICE_ACCOUNT,
    tensorboard = TENSORBOARD_RESOURCE_NAME,
    machine_type="a2-highgpu-1g",
    accelerator_type="NVIDIA_TESLA_A100",
    accelerator_count = 1,
)
```

## Optional: Finetuning larger models with A100 80 GB

[`google/flan-t5-base`](https://huggingface.co/google/flan-t5-base) has 247577856 parameters. In case of using a larger version of Flan-T5, for example, [`google/flan-t5-large`](https://huggingface.co/google/flan-t5-large) you must use a larger A100 with 80 GB HBM. For even bigger versions, you must use distributed training with multiple GPUs, or TPUs.

In order to use the [NVIDIA A100 80GB](https://cloud.google.com/vertex-ai/docs/training/configure-compute#specifying_gpus), you must use the `gcloud beta ai` command or the `v1beta1 Vertex AI API`. As of today (February 2023), A100 80GB is [available](https://cloud.google.com/compute/docs/gpus/gpu-regions-zones) in two US regions: `us-central1-c` and `us-east4-c`.


## Uvicorn

A Custom Container image for  predictions is required. Custom Container image [requires that the container must run an HTTP server](https://cloud.google.com/ai-platform-unified/docs/predictions/custom-container-requirements#image). 
Specifically, the container must listen and respond to liveness checks, health checks, and prediction requests.

This repo uses **FastAPI and Uvicorn** to implement the HTTP server. 
The HTTP server must listen for requests on `0.0.0.0`. [Uvicorn](https://www.uvicorn.org) is an ASGI web server implementation for Python. 
Uvicorn currently supports HTTP/1.1 and WebSockets. 
[Here](https://github.com/tiangolo/uvicorn-gunicorn-fastapi-docker) is a docker image with Uvicorn managed by Gunicorn for high-performance FastAPI web applications in Python 3.6+ with performance auto-tuning. 
An uvicorn server is launched with:
```bash
uvicorn main:app --host 0.0.0.0 --port 8080
```


## Download model

After finetuning, the model is stored in GCS. Download the resulting model to the local `predict/model-output-flan-t5-base` directory (for example, with `gcloud storage cp`). The model will be embedded inside the Custom Prediction Container. The `predict/` directory contains the `Dockerfile` to generate the Custom Prediction Container image. 

The model must be available in the `predict/model-output-flan-t5-base` directory, with a similar content like this, before building the custom container prediction image:
```sh
-rw-r--r--   1 rafaelsanchez  primarygroup       1399 14 Sep 11:19 config.json
-rw-r--r--   1 rafaelsanchez  primarygroup  304670597 14 Sep 11:19 pytorch_model.bin
-rw-r--r--   1 rafaelsanchez  primarygroup     839750 14 Sep 11:19 source.spm
-rw-r--r--   1 rafaelsanchez  primarygroup         65 14 Sep 11:19 special_tokens_map.json
-rw-r--r--   1 rafaelsanchez  primarygroup     796647 14 Sep 11:19 target.spm
-rw-r--r--   1 rafaelsanchez  primarygroup        296 14 Sep 11:19 tokenizer_config.json
-rw-r--r--   1 rafaelsanchez  primarygroup       3183 14 Sep 11:19 training_args.bin
-rw-r--r--   1 rafaelsanchez  primarygroup    1688744 14 Sep 11:19 vocab.json
```


## Build Custom Prediction Container image and upload model to Vertex AI Model Registry

Push docker image to **Artifact Registry**:
```sh
gcloud auth configure-docker europe-west4-docker.pkg.dev
gcloud builds submit --tag europe-west4-docker.pkg.dev/argolis-rafaelsanchez-ml-dev/ml-pipelines-repo/finetuning_flan_t5_base
```

Upload model to Vertex AI Prediction using the previous image with:
```sh
python3 upload_custom.py
```

```python
DEPLOY_IMAGE = 'europe-west4-docker.pkg.dev/argolis-rafaelsanchez-ml-dev/ml-pipelines-repo/finetuning_flan_t5_base' 
HEALTH_ROUTE = "/health"
PREDICT_ROUTE = "/predict"
SERVING_CONTAINER_PORTS = [8080]

model = aiplatform.Model.upload(
    display_name=f'custom-finetuning_flan_t5_base',    
    description=f'Finetuned Flan T5 model with Uviron and FastAPI',
    serving_container_image_uri=DEPLOY_IMAGE,
    serving_container_predict_route=PREDICT_ROUTE,
    serving_container_health_route=HEALTH_ROUTE,
    serving_container_ports=SERVING_CONTAINER_PORTS,
)

# Retrieve a Model on Vertex
model = aiplatform.Model(model.resource_name)

# Deploy model
endpoint = model.deploy(
    machine_type='a2-highgpu-1g',
    traffic_split={"0": 100}, 
    min_replica_count=1,
    max_replica_count=1,
    accelerator_type= "NVIDIA_TESLA_A100",    
    accelerator_count=1,
    traffic_percentage=100,
    deploy_request_timeout=1200,
    sync=True,
)
endpoint.wait()
```


## Online predictions

Predict using the Vertex AI Python SDK with `python3 predict_cloud.py`:
```py
# Retrieve an Endpoint on Vertex
print(endpoint.predict([[sample["dialogue"]]]))
# Output: 
# Prediction(predictions=[[["Patti's cat is fine. Patti will pick her up later. Patti will fetch the cage after work."]]], 
# deployed_model_id='4495568794441220096', model_version_id='1', model_resource_name='projects/989788194604/locations/europe-west4/models/2128205910630203392', explanations=None)

```

Predict using Vertex AI REST API:
```bash
curl -X POST -H "Authorization: Bearer $(gcloud auth print-access-token)" -H "Content-Type: application/json" \
https://europe-west4-aiplatform.googleapis.com/v1/projects/989788194604/locations/europe-west4/endpoints/7028544517674369024:predict \
 \
-d "{\"instances\": [\"
Greg: Hi Mum, how's the cat doing?
Patti: I just rang the vets, she's fine!
Greg: Thank God, been worrying about her all day!
Patti: They said I can pick her up later. I'll pop home and fetch the cage after work. Should be there at 5ish.
Greg: Good, see you at home, bye!
\"]}"
# Output
{
  "predictions": [
    [
      [
        "Patti will pick up the cat later. Patti will pop home and fetch the cage after work."
      ]
    ]
  ],
  "deployedModelId": "4495568794441220096",
  "model": "projects/989788194604/locations/europe-west4/models/2128205910630203392",
  "modelDisplayName": "custom-finetuning-flan-t5-base",
  "modelVersionId": "1"
}
```


## References

[1] Phil Schmid blog: [Fine-tune FLAN-T5 for chat & dialogue summarization](https://www.philschmid.de/fine-tune-flan-t5)    



