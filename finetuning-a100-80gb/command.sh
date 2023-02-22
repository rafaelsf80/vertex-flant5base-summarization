""" In case of using NVIDIA A100 80 GB for fine tuning, only US regions and 'gcloud beta ai' is supported """

gcloud beta ai custom-jobs create \
--region=us-central1 \
--display-name=fine_tune_flant5large \
--worker-pool-spec=machine-type=a2-ultragpu-1g,replica-count=1,accelerator-type=NVIDIA_A100_80GB,container-image-uri="us-central1-docker.pkg.dev/argolis-rafaelsanchez-ml-dev/ml-pipelines-repo/finetuning_flan_t5_large"



