from kubernetes.client import models as k8s

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from airflow.decorators import dag, task
from airflow.operators.trigger_dagrun import TriggerDagRunOperator


IMAGE='harbor.neoflex.ru/dognauts/dognauts-airflow:2.5.3-py3.8-v7-cicd'
EXECUTOR_CONFIG = {"pod_override": k8s.V1Pod(spec=k8s.V1PodSpec(containers=[k8s.V1Container(name="base", image=IMAGE)]))}


DB_URI = f"postgresql://postgres:{os.environ.get('DB_PASSWORD', 'postgres')}@cassandra-postgresql.feast-db:5432/FEAST_OFFLINE_STORE"
S3_OPTIONS = {'key': os.getenv('AWS_ACCESS_KEY_ID'), 
            'secret': os.getenv('AWS_SECRET_ACCESS_KEY'), 
            'client_kwargs': {'endpoint_url': os.getenv('KUBERNETES_MLFLOW_S3_ENDPOINT_URL')}}
BUCKET = os.getenv('KUBERNETES_S3_BUCKET')


RANDOM_SEED = 42


@dag('data_preprocessing_dag',\
            schedule_interval=None,
            default_args={
                "owner": "nsafonov",
                "retries": 1,
                "retry_delay": timedelta(minutes=5),
                "start_date": datetime(2021, 1, 1),
            })
def train():

    @task(task_id='data_preprocess', executor_config = EXECUTOR_CONFIG)
    def preprocess_data():
        from sklearn.preprocessing import LabelEncoder

        df_train = pd.read_sql('select * from train', con = DB_URI)
        df_test = pd.read_sql('select * from test', con = DB_URI)

        df_train['Train'] = 1
        df_test['Train'] = 0

        df = pd.concat([df_train, df_test], ignore_index=True)

        datetime_cols = ['app_date']
        bin_cols = ['sex', 'car', 'car_type', 'good_work', 'foreign_passport']
        cat_cols = ['education', 'region_rating', 'home_address', 'work_address', 'sna', 'first_time']
        num_cols = ['age', 'decline_app_cnt', 'bki_request_cnt', 'income']

        # scaling numeric columns
        for i in num_cols:
            df[i] = np.log(df[i] + 1)

        # encoding categorical columns
        for column in bin_cols:
            df[column] = LabelEncoder().fit_transform(df[column])

        # filling nans in education
        df['education'] = df['education'].fillna('SCH')

        # extracting days from app_date
        df.app_date = pd.to_datetime(df.app_date, format='%d%b%Y')
        start = df.app_date.min()
        df['days'] = (df.app_date - start).dt.days.astype('int')

        # one-hot encoding categorical columns
        df = pd.get_dummies(df, prefix=cat_cols, columns=cat_cols)

        data = df.drop(columns = 'app_date')
        
        data.to_parquet(f's3://{BUCKET}/airflow/preprocessed_data/dataset.parquet', \
                        index=False,\
                        storage_options=S3_OPTIONS)
    
    
    @task(task_id='data_split', executor_config = EXECUTOR_CONFIG)
    def splitting_train_data():
        from sklearn.model_selection import train_test_split

        df = pd.read_parquet(f's3://{BUCKET}/airflow/preprocessed_data/dataset.parquet',
                            storage_options=S3_OPTIONS)

        train_df = df[df['Train'] == 1].drop(columns=['Train', 'client_id'])
        test_df = df[df['Train'] == 0].drop(columns=['Train', 'client_id'])

        x, y = train_df.drop(columns='default'), train_df[['default']]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_SEED)
        
        x_train.to_parquet(f's3://{BUCKET}/airflow/train_data/x_train.parquet', index = False, storage_options = S3_OPTIONS)
        x_test.to_parquet(f's3://{BUCKET}/airflow/train_data/x_test.parquet', index = False, storage_options = S3_OPTIONS)
        y_train.to_parquet(f's3://{BUCKET}/airflow/train_data/y_train.parquet', index = False, storage_options = S3_OPTIONS)
        y_test.to_parquet(f's3://{BUCKET}/airflow/train_data/y_test.parquet', index = False, storage_options = S3_OPTIONS)
               
    trigger = TriggerDagRunOperator(
        task_id="trigger_model_training",
        trigger_dag_id="model_training_dag",
    )

    preprocess_data() >> splitting_train_data() >> trigger

train()
