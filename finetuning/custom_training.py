""" Custom training pipeline, with local script located at 'flant5large_trainer.py' """

from google.cloud import aiplatform

BUCKET = 'gs://argolis-vertex-europewest4'
PROJECT_ID = 'argolis-rafaelsanchez-ml-dev'
LOCATION = 'europe-west4'
SERVICE_ACCOUNT = 'tensorboard-sa@argolis-rafaelsanchez-ml-dev.iam.gserviceaccount.com'
TENSORBOARD_RESOURCE_NAME = 'projects/989788194604/locations/europe-west4/tensorboards/8884581718011412480'

# Initialize the *client* for Vertex
aiplatform.init(project=PROJECT_ID, staging_bucket=BUCKET, location=LOCATION)
                
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