
from kubernetes.client import models as k8s

from airflow.decorators import dag, task
import pandas as pd
from datetime import datetime, timedelta
import os

MODEL_NAME = "rfc_model"
INPUT_FILE = "airflow/train_data/x_test.parquet"
OUTPUT_FILE = "airflow/inference_data/x_test_w_preds.parquet"


S3_OPTIONS = {
    'key': os.getenv('AWS_ACCESS_KEY_ID'),
    'secret': os.getenv('AWS_SECRET_ACCESS_KEY'),
    'client_kwargs': {'endpoint_url': os.getenv('KUBERNETES_MLFLOW_S3_ENDPOINT_URL')},
}
BUCKET = os.getenv('KUBERNETES_S3_BUCKET')


IMAGE='harbor.neoflex.ru/dognauts/dognauts-airflow:2.5.3-py3.8-v7-cicd'

EXECUTOR_CONFIG = {"pod_override": k8s.V1Pod(spec=k8s.V1PodSpec(containers=[k8s.V1Container(name="base", image=IMAGE)]))}

@dag(
    'model_inference_dag',
    schedule_interval=None,
    default_args={
        "owner": "nsafonov",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
        "start_date": datetime(2021, 1, 1),
    },
)
def inference_dag():

    @task(task_id= 'default_prediction', executor_config = EXECUTOR_CONFIG)
    def inference():
        import mlflow

        models = mlflow.search_registered_models(filter_string=f'name = "{MODEL_NAME}"')
        last_version = max([int(i.version) for i in models[0].latest_versions if i.current_stage == 'Production'])

        model = mlflow.pyfunc.load_model(f'models:/{MODEL_NAME}/{last_version}')

        data = pd.read_parquet(f's3://{BUCKET}/{INPUT_FILE}', storage_options=S3_OPTIONS)

        predictions = model.predict(data)

        pd.DataFrame(predictions, columns=['pred']).to_parquet(
            f's3://{BUCKET}/{OUTPUT_FILE}', storage_options=S3_OPTIONS
        )

    inference()


inference_dag()
