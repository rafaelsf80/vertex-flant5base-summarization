from random import randrange

from datasets import load_dataset

from google.cloud import aiplatform

dataset_id = "samsum"

# Load dataset from the hub
dataset = load_dataset(dataset_id)

# select a random test sample
sample = dataset['test'][randrange(len(dataset["test"]))]
print(f"dialogue: \n{sample['dialogue']}\n---------------")

# Call endpoint
endpoint = aiplatform.Endpoint('projects/989788194604/locations/europe-west4/endpoints/7028544517674369024')
print(f"flan-t5-base summary:") 
print(endpoint.predict([[sample["dialogue"]]]))
# Output: 
# Prediction(predictions=[[["Patti's cat is fine. Patti will pick her up later. Patti will fetch the cage after work."]]], 
# deployed_model_id='4495568794441220096', model_version_id='1', model_resource_name='projects/989788194604/locations/europe-west4/models/2128205910630203392', explanations=None)


