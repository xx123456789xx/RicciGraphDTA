import os
import sys
import csv
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from torch.utils.data import DataLoader
import torch.nn as nn

# 模型导入
from models.GINConvNetWithCurvature import GINConvNetWithCurvature
from utils import *

# ---------- 评估指标函数 ----------
def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs); y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for _ in y_obs]
    y_pred_mean = [np.mean(y_pred) for _ in y_pred]
    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult
    y_obs_sq = sum((y_obs - y_obs_mean) ** 2)
    y_pred_sq = sum((y_pred - y_pred_mean) ** 2)
    return mult / float(y_obs_sq * y_pred_sq)

def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs); y_pred = np.array(y_pred)
    return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))

def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)
    y_obs = np.array(y_obs); y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for _ in y_obs]
    upp = sum((y_obs - (k * y_pred)) ** 2)
    down = sum((y_obs - y_obs_mean) ** 2)
    return 1 - (upp / float(down))

def get_rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)
    return r2 * (1 - np.sqrt(abs((r2 * r2) - (r02 * r02))))

# ---------- 训练函数 ----------
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        bs = data.x.shape[0]
        total_loss += loss.item() * bs
        total_samples += bs
        if batch_idx % LOG_INTERVAL == 0:
            print(f'Train epoch: {epoch} [{batch_idx * bs}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    avg_loss = total_loss / total_samples
    print(f"Epoch {epoch} average training loss: {avg_loss:.6f}")
    return avg_loss

# ---------- 修改后的预测函数 ----------
def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    all_smiles = []
    all_prots = []
    
    print(f'Make prediction for {len(loader.dataset)} samples...')
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
            
            # 修改这里：使用 protein_sequence 而不是 protein_id
            if hasattr(data, 'protein_sequence'):
                all_prots.extend(data.protein_sequence)
            elif hasattr(data, 'target'):
                all_prots.extend(data.target)
            else:
                raise AttributeError("Data object has no protein information field")
                
            if hasattr(data, 'smiles'):
                all_smiles.extend(data.smiles)
            else:
                raise AttributeError("Data object has no smiles field")
    
    return (total_labels.numpy().flatten(),
            total_preds.numpy().flatten(),
            all_smiles,
            all_prots)

# ---------- 主程序 ----------
if __name__ == '__main__':
    # 解析命令行参数
    datasets = [['davis', 'kiba'][int(sys.argv[1])]]
    modeling = [GINConvNetWithCurvature][int(sys.argv[2])]
    model_st = modeling.__name__

    cuda_name = "cuda:0" if len(sys.argv) <= 3 else f"cuda:{int(sys.argv[3])}"
    print('Using device:', cuda_name)

    # 超参数设置
    TRAIN_BATCH_SIZE = 512
    TEST_BATCH_SIZE = 256
    LR = 0.0005
    LOG_INTERVAL = 20
    NUM_EPOCHS = 1000

    # 初始化
    os.makedirs('plots', exist_ok=True)
    csv_file = f'training_results_{model_st}_{datasets[0]}.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch','Loss','RMSE','MSE','CI','RM2'])
        writer.writeheader()

    # 训练循环
    for dataset in datasets:
        print(f'\n>> Running on {model_st}_{dataset}')
        
        # 数据加载
        train_data = TestbedDataset(root='data', dataset=f'{dataset}_train')
        test_data = TestbedDataset(root='data', dataset=f'{dataset}_test')
        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

        # 模型初始化
        device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
        model = modeling().to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        best_mse, best_epoch = float('inf'), -1

        for epoch in range(1, NUM_EPOCHS + 1):
            # 训练
            avg_loss = train(model, device, train_loader, optimizer, epoch)

            # 评估
            G, P, smiles_list, prot_list = predicting(model, device, test_loader)
            current_rmse = rmse(G, P)
            current_mse = mse(G, P)
            current_ci = ci(G, P)
            current_rm2 = get_rm2(G, P)

            print(f'Epoch {epoch} - Loss: {avg_loss:.6f}, '
                  f'RMSE: {current_rmse:.4f}, MSE: {current_mse:.4f}, '
                  f'CI: {current_ci:.4f}, RM²: {current_rm2:.4f}')

            # 记录结果
            with open(csv_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['epoch','Loss','RMSE','MSE','CI','RM2'])
                writer.writerow({
                    'epoch': epoch,
                    'Loss': avg_loss,
                    'RMSE': current_rmse,
                    'MSE': current_mse,
                    'CI': current_ci,
                    'RM2': current_rm2
                })

            # 保存异常值（采用第二个代码示例的方法）
            if current_mse < best_mse:
                errors = np.abs(G - P)
                top_n = 10
                top_idx = errors.argsort()[::-1][:top_n]
                
                # 创建DataFrame保存异常值信息
                df_abnormal = pd.DataFrame({
                    'index': top_idx,
                    'SMILES': [smiles_list[i] for i in top_idx],
                    'Protein': [prot_list[i] for i in top_idx],
                    'True': G[top_idx],
                    'Predicted': P[top_idx],
                    'Error': errors[top_idx]
                })
                
                # 保存到CSV文件
                abnormal_file = f'abnormal_cases_{model_st}_{dataset}.csv'
                df_abnormal.to_csv(abnormal_file, index=False)
                print(f'Saved abnormal cases to {abnormal_file}')

                # 保存最佳模型
                torch.save(model.state_dict(), f'model_{model_st}_{dataset}.model')
                best_mse, best_epoch = current_mse, epoch
                print(f'  >> New best MSE {best_mse:.4f} at epoch {best_epoch}')
            else:
                print(f'  -- No improvement since epoch {best_epoch}')

    print("Training complete.")