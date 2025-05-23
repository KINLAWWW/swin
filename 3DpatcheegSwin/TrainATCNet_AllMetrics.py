import torch
import torch.nn as nn
from strokes import StrokePatientsMIDataset, StrokePatientsMIProcessedDataset
from strokesdict import STROKEPATIENTSMI_LOCATION_DICT
from torcheeg.transforms import Select,BandSignal,Compose
from to import ToTensor
from downsample import SetSamplingRate
from baseline import BaselineCorrection
import os
import pandas as pd
import numpy as np
from torcheeg.model_selection import KFoldPerSubjectGroupbyTrial
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader

dataset = StrokePatientsMIDataset(root_path='./subdataset',
                                  io_path='.torcheeg/dataset',
                        chunk_size=500,  # 1 second
                        overlap = 250,
                        offline_transform=Compose(
                                [BaselineCorrection(),
                                SetSamplingRate(origin_sampling_rate=500,target_sampling_rate=128),
                                BandSignal(sampling_rate=500,band_dict={'frequency_range':[8,40]})
                                ]),
                        online_transform=Compose(
                                [ToTensor()]),
                        label_transform=Select('label'),
                        num_worker=8
)

from torcheeg.models import ATCNet

HYPERPARAMETERS = {
    "seed": 42,
    "batch_size": 16,
    "lr": 1e-4,
    "weight_decay": 0,
    "num_epochs": 200,
}
from torcheeg.model_selection import KFoldPerSubjectGroupbyTrial
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from classifier import ClassifierTrainer

k_fold = KFoldPerSubjectGroupbyTrial(n_splits=4, shuffle=True,split_path='.torcheeg/atcnet_model_selection',random_state=42)

metrics = ['accuracy', 'recall', 'precision', 'f1score','kappa']
csv_path = 'logs/ATCNet_testmetrics_results.csv'

# 创建文件夹
os.makedirs(os.path.dirname(csv_path), exist_ok=True)


for i, (training_dataset, test_dataset) in enumerate(k_fold.split(dataset)):
    print(f"开始训练 Fold {i + 1}: Train size = {len(training_dataset)}, Test size = {len(test_dataset)}", flush=True)
    model = ATCNet(in_channels=1,
                    num_classes=2,
                    num_windows=2,
                    num_electrodes=30,
                    chunk_size=128)
    trainer = ClassifierTrainer(model=model,
                                num_classes=2,
                                lr=HYPERPARAMETERS['lr'],
                                weight_decay=HYPERPARAMETERS['weight_decay'],
                                metrics=['accuracy', 'recall', 'precision', 'f1score', 'kappa'],
                                accelerator="gpu")
    training_loader = DataLoader(training_dataset,
                            batch_size=HYPERPARAMETERS['batch_size'],
                            shuffle=True)
    test_loader = DataLoader(test_dataset,
                            batch_size=HYPERPARAMETERS['batch_size'],
                            shuffle=False)
    # 提前停止回调
    early_stopping_callback = EarlyStopping(
        monitor='train_loss',
        patience=50,
        mode='min',
        verbose=True
    )
    trainer.fit(training_loader,
                test_loader,
                max_epochs=HYPERPARAMETERS['num_epochs'],
                callbacks=[early_stopping_callback],
                enable_progress_bar=False,
                enable_model_summary=False,
                limit_val_batches=0.0)
    training_result = trainer.test(training_loader,
                                enable_progress_bar=False,
                                enable_model_summary=True)[0]
    test_result = trainer.test(test_loader,
                            enable_progress_bar=False,
                            enable_model_summary=True)[0]

    # 构造字典
    row = {metric: test_result[f"test_{metric}"] for metric in metrics}
    row["fold"] = i + 1  # 添加 fold 编号

    # 调整列顺序：fold 放第一列
    columns_order = ["fold"] + metrics
    row_df = pd.DataFrame([row])[columns_order]

    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        df_metrics = pd.concat([existing, row_df], ignore_index=True)
    else:
        df_metrics = row_df

    df_metrics.to_csv(csv_path, index=False, float_format='%.4f')


# 加载原始 CSV
df = pd.read_csv(csv_path)

# 去除非数值列（如果有 fold 或 run 列）
numeric_df = df.select_dtypes(include='number')

# 每 4 行为一组，取平均
group_size = 4
grouped = numeric_df.groupby(numeric_df.index // group_size).mean()

# 添加 group 编号列（从 1 开始）
grouped.insert(0, 'group', grouped.index + 1)

# 保存新结果
grouped.to_csv('logs/ATCNet_testmetrics_results_AVG.csv', index=False, float_format='%.4f')
print("✅ 每 4 行平均结果已保存到 logs/ATCNet_testmetrics_results_AVG.csv")

