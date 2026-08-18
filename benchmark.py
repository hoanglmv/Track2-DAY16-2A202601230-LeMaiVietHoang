import os
import sys
import time
import json
import urllib.request
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

DATA_DIR = os.path.expanduser('~/ml-benchmark')
CSV_PATH = os.path.join(DATA_DIR, 'creditcard.csv')
JSON_PATH = os.path.join(DATA_DIR, 'benchmark_result.json')

os.makedirs(DATA_DIR, exist_ok=True)

print('=== 1. Loading Dataset ===')
if not os.path.exists(CSV_PATH):
    print(f'File {CSV_PATH} not found. Downloading dataset...')
    mirror_url = 'https://raw.githubusercontent.com/nethal/Credit-Card-Fraud-Detection/master/creditcard.csv'
    try:
        urllib.request.urlretrieve(mirror_url, CSV_PATH)
        print(f'Downloaded dataset to {CSV_PATH}')
    except Exception as e:
        print(f'Download error: {e}')
        sys.exit(1)

start_load = time.time()
df = pd.read_csv(CSV_PATH)
load_time = time.time() - start_load
print(f'Data shape: {df.shape}')
print(f'Data load time: {load_time:.4f} seconds')

X = df.drop(columns=['Class'])
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print('\n=== 2. Training LightGBM Model ===')
start_train = time.time()
model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(50, verbose=False)]
)
train_time = time.time() - start_train
best_iter = getattr(model, 'best_iteration_', 100) or 100
print(f'Training completed in {train_time:.4f} seconds')
print(f'Best iteration: {best_iter}')

print('\n=== 3. Evaluating Model ===')
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

auc_roc = float(roc_auc_score(y_test, y_pred_proba))
accuracy = float(accuracy_score(y_test, y_pred))
f1 = float(f1_score(y_test, y_pred))
precision = float(precision_score(y_test, y_pred))
recall = float(recall_score(y_test, y_pred))

print('\n=== 4. Benchmarking Inference ===')
sample_1_row = X_test.iloc[[0]]
for _ in range(10):
    model.predict_proba(sample_1_row)

n_latency_iters = 500
t0 = time.time()
for _ in range(n_latency_iters):
    model.predict_proba(sample_1_row)
latency_ms = ((time.time() - t0) / n_latency_iters) * 1000.0

sample_1000_rows = X_test.iloc[:1000]
for _ in range(5):
    model.predict_proba(sample_1000_rows)

n_throughput_iters = 50
t0 = time.time()
for _ in range(n_throughput_iters):
    model.predict_proba(sample_1000_rows)
batch_time_sec = (time.time() - t0) / n_throughput_iters
throughput_qps = 1000.0 / batch_time_sec

results = {
    'data_load_time_sec': round(load_time, 4),
    'training_time_sec': round(train_time, 4),
    'best_iteration': int(best_iter),
    'auc_roc': round(auc_roc, 6),
    'accuracy': round(accuracy, 6),
    'f1_score': round(f1, 6),
    'precision': round(precision, 6),
    'recall': round(recall, 6),
    'inference_latency_1_row_ms': round(latency_ms, 4),
    'inference_throughput_1000_rows_qps': round(throughput_qps, 2)
}

with open(JSON_PATH, 'w') as f:
    json.dump(results, f, indent=4)

print('\n==================================================')
print('             LIGHTGBM BENCHMARK RESULTS           ')
print('==================================================')
print(f'Thời gian load data       : {results["data_load_time_sec"]} s')
print(f'Thời gian training        : {results["training_time_sec"]} s')
print(f'Best iteration            : {results["best_iteration"]}')
print(f'AUC-ROC                   : {results["auc_roc"]}')
print(f'Accuracy                  : {results["accuracy"]}')
print(f'F1-Score                  : {results["f1_score"]}')
print(f'Precision                 : {results["precision"]}')
print(f'Recall                    : {results["recall"]}')
print(f'Inference latency (1 row) : {results["inference_latency_1_row_ms"]} ms')
print(f'Throughput (1000 rows)    : {results["inference_throughput_1000_rows_qps"]} rows/sec')
print('==================================================')
print(f'Saved results to: {JSON_PATH}')
