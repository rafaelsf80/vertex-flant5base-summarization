# gcloud builds submit --tag europe-west4-docker.pkg.dev/argolis-rafaelsanchez-ml-dev/ml-pipelines-repo/finetuning_flan_t5_base
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

# install dependencies
RUN python3 -m pip install --upgrade pip
RUN pip3 install transformers==4.25.1 sentencepiece==0.1.97 torch==1.13.0

COPY ./app /app
#COPY requirements.txt requirements.txt
COPY ./model-output-flan-t5-base/ /model-output-flan-t5-base/

#RUN pip install -r requirements.txt
EXPOSE 8080

# Start the app
CMD ["gunicorn", "-b", "0.0.0.0:8080", "main:app","--workers","1","--timeout","180","-k","uvicorn.workers.UvicornWorker"]