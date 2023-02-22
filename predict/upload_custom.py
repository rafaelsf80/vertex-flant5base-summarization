from google.cloud import aiplatform

STAGING_BUCKET = 'gs://argolis-vertex-europewest4'
PROJECT_ID = 'argolis-rafaelsanchez-ml-dev'
LOCATION = 'europe-west4'

aiplatform.init(project=PROJECT_ID, staging_bucket=STAGING_BUCKET, location=LOCATION)

DEPLOY_IMAGE = 'europe-west4-docker.pkg.dev/argolis-rafaelsanchez-ml-dev/ml-pipelines-repo/finetuning_flan_t5_base' 
HEALTH_ROUTE = "/health"
PREDICT_ROUTE = "/predict"
SERVING_CONTAINER_PORTS = [8080]

model = aiplatform.Model.upload(
    display_name=f'custom-finetuning-flan-t5-base',    
    description=f'Finetuned Flan T5 model with Uviron and FastAPI',
    serving_container_image_uri=DEPLOY_IMAGE,
    serving_container_predict_route=PREDICT_ROUTE,
    serving_container_health_route=HEALTH_ROUTE,
    serving_container_ports=SERVING_CONTAINER_PORTS,
)

print(model.resource_name)

# Retrieve a Model on Vertex
model = aiplatform.Model(model.resource_name)
print(model.resource_name)

# Deploy model
endpoint = model.deploy(
      machine_type='n1-standard-4', 
      sync=False
)
endpoint.wait()
# endpoint = model.deploy(
#     machine_type='a2-highgpu-1g',
#     traffic_split={"0": 100}, 
#     min_replica_count=1,
#     max_replica_count=1,
#     accelerator_type= "NVIDIA_TESLA_A100",    
#     accelerator_count=1,
#     traffic_percentage=100,
#     deploy_request_timeout=1200,
#     sync=True,
# )
# endpoint.wait()