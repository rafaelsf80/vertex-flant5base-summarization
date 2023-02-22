# gcloud builds submit --tag us-central1-docker.pkg.dev/argolis-rafaelsanchez-ml-dev/ml-pipelines-repo/finetuning_flan_t5_large
FROM europe-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-10:latest

# set working directory
WORKDIR /app

# Install required packages
RUN pip install py7zr==0.20.4 nltk==3.7 evaluate==0.4.0 rouge_score==0.1.2 transformers==4.25.1 tensorboard==2.11.2 datasets==2.9.0 google-cloud-storage==2.7.0

# Copies the trainer code to the docker image.
COPY ./trainer/task.py /app/trainer/task.py

# Set up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "trainer.task"]