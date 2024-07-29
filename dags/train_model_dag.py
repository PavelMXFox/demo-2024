from kubernetes.client import models as k8s

try:
    from airflow import DAG
    from airflow.operators.python_operator import PythonOperator
    from airflow.decorators import task
    
    from datetime import datetime, timedelta
    import pandas as pd
    import os

except Exception as e:
    print("Error  {} ".format(e))

#IMAGE='registry.neomsa.ru/docker-mlops/mlops/airflow:executor-13'
#IMAGE='registry.neomsa.ru/docker-mlops/mlops/airflow255:executor-2'
#IMAGE='registry.neomsa.ru/docker-mlops/mlops/airflow:2.2.5-demo2'
IMAGE='harbor-dognauts.neoflex.ru/dognauts/dognauts-airflow:2.5.3-py3.8-v6'

RANDOM_SEED = 42

def StandardScaler_df_and_filna_0(d_df, d_columns):
    for i  in list(d_df[d_columns].columns):
        d_df[i] = StandardScaler_column(d_df, i)
        if len(d_df[d_df[i].isna()]) < len(d_df):
            d_df[i] = d_df[i].fillna(d_df[i].min())
    return

def prepare_data():
    from sqlalchemy import create_engine
    from sqlalchemy.engine.url import URL
    import boto3
    import pandas as pd
    import os
    import numpy as np
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    def StandardScaler_column(d_df, d_col):
        scaler = StandardScaler()
        scaler.fit(d_df[[d_col]])
        return scaler.transform(d_df[[d_col]])

    def StandardScaler_df_and_filna_0(d_df, d_columns):
        for i  in list(d_df[d_columns].columns):
            d_df[i] = StandardScaler_column(d_df, i)
            if len(d_df[d_df[i].isna()]) < len(d_df):
                d_df[i] = d_df[i].fillna(d_df[i].min())
        return

    database_password = os.environ.get('DB_PASSWORD', 'postgres')
    bucket_name = os.environ.get('KUBERNETES_S3_BUCKET')
    s3_endpoint = os.environ.get('KUBERNETES_MLFLOW_S3_ENDPOINT_URL')
    access_key = os.environ.get('AWS_ACCESS_KEY_ID')
    secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')

    DATABASE = {
        'drivername': 'postgresql+psycopg2',
        'host': 'cassandra-postgresql.feast-db',
        'port': '5432',
        'username': 'postgres',
        'password': database_password,
        'database': 'FEAST_OFFLINE_STORE'
    }

    from sqlalchemy import create_engine
    from sqlalchemy.engine.url import URL
    engine  = create_engine(URL(**DATABASE))

    df_train = pd.read_sql_query('SELECT * FROM train',con=engine)
    df_test = pd.read_sql_query('SELECT * FROM test',con=engine)

    df_train['Train'] = 1
    df_test['Train'] = 0

    df = df_train.append(df_test, sort=False).reset_index(drop=True)

    time_cols = ['app_date']
    bin_cols = ['sex', 'car', 'car_type', 'good_work', 'foreign_passport']
    cat_cols = ['education', 'region_rating', 'home_address', 'work_address', 'sna', 'first_time']
    num_cols = ['age','decline_app_cnt','score_bki','bki_request_cnt','income','days']

    df['age'] = np.log(df['age'] + 1)
    df['decline_app_cnt'] = np.log(df['decline_app_cnt'] + 1)
    df['bki_request_cnt'] = np.log(df['bki_request_cnt'] + 1)
    df['income'] = np.log(df['income'] + 1)
    df['education'] = df['education'].fillna('SCH')
    df.app_date = pd.to_datetime(df.app_date, format='%d%b%Y')

    start = df.app_date.min()
    end = df.app_date.max()
    df['days'] = (df.app_date - start).dt.days.astype('int')

    label_encoder = LabelEncoder()
    df['education_l'] = label_encoder.fit_transform(df['education'])

    label_encoder = LabelEncoder()
    for column in bin_cols:
        df[column] = label_encoder.fit_transform(df[column])

    df=pd.get_dummies(df, prefix=cat_cols, columns=cat_cols)

    df.drop(['app_date', 'education_l'], axis=1, inplace=True)

    data = df.drop(['index'], axis=1)
    data.to_csv('/tmp/dataset.csv', sep='\t', encoding='utf-8', index=False)

    s3 = boto3.resource('s3',
                        endpoint_url=s3_endpoint,
                        aws_access_key_id=access_key,
                        aws_secret_access_key=secret_key)
    s3.meta.client.upload_file('/tmp/dataset.csv', Bucket=bucket_name, Key='airflow/dataset.csv')


def split_data():
    import pandas as pd
    import os
    import boto3
    from sklearn.model_selection import train_test_split

    bucket_name = os.environ.get('KUBERNETES_S3_BUCKET')
    s3_endpoint = os.environ.get('KUBERNETES_MLFLOW_S3_ENDPOINT_URL')
    access_key = os.environ.get('AWS_ACCESS_KEY_ID')
    secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')

    s3 = boto3.resource('s3',
                        endpoint_url=s3_endpoint,
                        aws_access_key_id=access_key,
                        aws_secret_access_key=secret_key)
    s3.meta.client.download_file(bucket_name, 'airflow/dataset.csv', '/tmp/dataset.csv')

    df = pd.read_csv('/tmp/dataset.csv', sep='\t', encoding='utf-8')

    train_data = df.query('Train == 1').drop(['Train', 'client_id'], axis=1)
    test_data = df.query('Train == 0').drop(['Train', 'client_id'], axis=1)

    y = train_data.default.values            # наш таргет
    X = train_data.drop(['default'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    
    y_train_df = pd.DataFrame(y_train, columns = ['fact'])
    X_train.to_csv('/tmp/train_x.csv', sep='\t', encoding='utf-8', index=False)
    y_train_df.to_csv('/tmp/train_y.csv', sep='\t', encoding='utf-8', index=False)
    X_test.to_csv('/tmp/X_test.csv', sep='\t', encoding='utf-8', index=False)
    y_test_df = pd.DataFrame(y_test, columns = ['fact'])
    y_test_df.to_csv('/tmp/y_test.csv', sep='\t', encoding='utf-8', index=False)

    s3.meta.client.upload_file('/tmp/train_x.csv', Bucket=bucket_name, Key='airflow/train_x.csv')
    s3.meta.client.upload_file('/tmp/train_y.csv', Bucket=bucket_name, Key='airflow/train_y.csv')
    s3.meta.client.upload_file('/tmp/X_test.csv', Bucket=bucket_name, Key='airflow/X_test.csv')
    s3.meta.client.upload_file('/tmp/y_test.csv', Bucket=bucket_name, Key='airflow/y_test.csv')

def train_model():
    import pandas as pd
    import os
    import boto3
    import mlflow
    import matplotlib
    from sklearn.linear_model import LogisticRegression

    bucket_name = os.environ.get('KUBERNETES_S3_BUCKET')
    s3_endpoint = os.environ.get('KUBERNETES_MLFLOW_S3_ENDPOINT_URL')
    access_key = os.environ.get('AWS_ACCESS_KEY_ID')
    secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')

    s3 = boto3.resource('s3',
                        endpoint_url=s3_endpoint,
                        aws_access_key_id=access_key,
                        aws_secret_access_key=secret_key)
    s3.meta.client.download_file(bucket_name, 'airflow/train_x.csv', '/tmp/train_x.csv')
    s3.meta.client.download_file(bucket_name, 'airflow/train_y.csv', '/tmp/train_y.csv')
    s3.meta.client.download_file(bucket_name, 'airflow/X_test.csv', '/tmp/X_test.csv')
    s3.meta.client.download_file(bucket_name, 'airflow/y_test.csv', '/tmp/y_test.csv')

    X_train = pd.read_csv('/tmp/train_x.csv', sep='\t', encoding='utf-8')
    y_train = pd.read_csv('/tmp/train_y.csv', sep='\t', encoding='utf-8').fact.values
    X_test = pd.read_csv('/tmp/X_test.csv', sep='\t', encoding='utf-8')
    y_test = pd.read_csv('/tmp/y_test.csv', sep='\t', encoding='utf-8').fact.values

    model = LogisticRegression(random_state=RANDOM_SEED)

    model.fit(X_train, y_train)

    mlflow.set_experiment('default-predict') 

    artifact_path = "model"

    eval_data = X_test.copy()
    eval_data["target"] = y_test

    with mlflow.start_run() as run:
        model_info = mlflow.sklearn.log_model(model, "model")
        result = mlflow.evaluate(
            model_info.model_uri,
            eval_data,
            targets="target",
            model_type="classifier",
            dataset_name="default-predict",
            evaluators="default",
            evaluator_config={"explainability_nsamples": 1000},
        )


with DAG(
        dag_id="train_model_dag",
        schedule_interval=None,
        default_args={
            "owner": "ovchintsev",
            "retries": 1,
            "retry_delay": timedelta(minutes=5),
            "start_date": datetime(2021, 1, 1),
        },
        catchup=False) as f:

    task1 = PythonOperator(
        task_id="task1",
        python_callable=prepare_data,
        provide_context=True,
        executor_config = {
        "pod_override": k8s.V1Pod(spec=k8s.V1PodSpec(containers=[k8s.V1Container(name="base", image=IMAGE)]))
    },
    )

    task2 = PythonOperator(
        task_id="task2",
        python_callable=split_data,
        provide_context=True,
        executor_config = {
        "pod_override": k8s.V1Pod(spec=k8s.V1PodSpec(containers=[k8s.V1Container(name="base", image=IMAGE)]))
    },
    )

    task3 = PythonOperator(
        task_id="task3",
        python_callable=train_model,
        provide_context=True,
        executor_config = {
        "pod_override": k8s.V1Pod(spec=k8s.V1PodSpec(containers=[k8s.V1Container(name="base", image=IMAGE)]))
    },
    )

task1 >> task2 >> task3
