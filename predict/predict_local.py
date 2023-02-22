from transformers import pipeline
from random import randrange

from datasets import load_dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration

dataset_id = "samsum"

# Load dataset from the hub
dataset = load_dataset(dataset_id)

# select a random test sample
sample = dataset['test'][randrange(len(dataset["test"]))]
print(f"dialogue: \n{sample['dialogue']}\n---------------")

# Load fine tuned model in local directory
model_id = './model-output-flan-t5-base'
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load tokenizer
model_name = './model-output-flan-t5-base'
model = T5ForConditionalGeneration.from_pretrained(model_name) 

# using pipeline
#summarizer = pipeline("summarization", model="model-output-flan-t5-base")#, device=0) # no GPU
#res = summarizer(sample["dialogue"])
#print(f"flan-t5-base summary:\n{res[0]['summary_text']}")

outputs = []
# summarize dialogue
generated = model.generate(**tokenizer(sample["dialogue"], return_tensors="pt", padding=True), max_new_tokens=50)
outputs.append([tokenizer.decode(t, skip_special_tokens=True) for t in generated])  
print(f"flan-t5-base summary:\n{outputs}") 

