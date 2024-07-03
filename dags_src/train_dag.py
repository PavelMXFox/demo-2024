from kubernetes.client import models as k8s

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from airflow.decorators import task, dag
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

IMAGE='harbor.neoflex.ru/dognauts/dognauts-airflow:2.5.3-py3.8-v7-cicd'

RANDOM_SEED = 42


EXECUTOR_CONFIG = {"pod_override": k8s.V1Pod(spec=k8s.V1PodSpec(containers=[k8s.V1Container(name="base", image=IMAGE)]))}

S3_OPTIONS = {'key': os.getenv('AWS_ACCESS_KEY_ID'), 
            'secret': os.getenv('AWS_SECRET_ACCESS_KEY'), 
            'client_kwargs': {'endpoint_url': os.getenv('KUBERNETES_MLFLOW_S3_ENDPOINT_URL')}}
BUCKET = os.getenv('KUBERNETES_S3_BUCKET')


MODEL_NAME = 'rfc_model'

@dag('model_training_dag',\
        schedule_interval=None,
        default_args={
            "owner": "nsafonov",
            "retries": 1,
            "retry_delay": timedelta(minutes=5),
            "start_date": datetime(2021, 1, 1),
        })
def train():

    @task(task_id='model_training', executor_config = EXECUTOR_CONFIG)
    def train_model():
        import mlflow
        from sklearn.ensemble import RandomForestClassifier


        x_train = pd.read_parquet(f's3://{BUCKET}/airflow/x_train.parquet', storage_options = S3_OPTIONS)
        x_test = pd.read_parquet(f's3://{BUCKET}/airflow/x_test.parquet', storage_options = S3_OPTIONS)
        y_train = pd.read_parquet(f's3://{BUCKET}/airflow/y_train.parquet', storage_options = S3_OPTIONS)
        y_test = pd.read_parquet(f's3://{BUCKET}/airflow/y_test.parquet', storage_options = S3_OPTIONS)


        # model = LogisticRegression(random_state=RANDOM_SEED)
        model = RandomForestClassifier(random_state=RANDOM_SEED)
        model.fit(x_train, y_train['default'].values)

        mlflow.set_experiment('default-predict') 

        eval_df = pd.concat([x_test, y_test], axis=1)

        mlflow.set_experiment("default_demo")

        with mlflow.start_run() as run:
            model_info = mlflow.sklearn.log_model(model, MODEL_NAME)
            mlflow.evaluate( 
                model_info.model_uri,
                eval_df,
                targets="default",
                model_type="classifier",
                evaluators="default",
                evaluator_config={"explainability_nsamples": 1000},
            )
            mv = mlflow.register_model(model_info.model_uri, MODEL_NAME)
            client = mlflow.MlflowClient()
            client.transition_model_version_stage(name=MODEL_NAME, \
                                                    version=mv.version, \
                                                    stage="Production", \
                                                    archive_existing_versions=True)

    trigger = TriggerDagRunOperator(
        task_id="trigger_model_inference",
        trigger_dag_id="model_inference_dag",
    )

    train_model() >> trigger

train()
