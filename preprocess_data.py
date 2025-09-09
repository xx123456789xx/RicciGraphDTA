import os
import json
import pickle
import numpy as np
import pandas as pd
from collections import OrderedDict
from rdkit import Chem
import networkx as nx
from torch_geometric.data import Data
from utils import TestbedDataset  # 此模块中 GCNData 定义已包含 protein_sequence 字段

# ---------- One-hot 编码工具 ----------
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"input {x} not in allowable set {allowable_set}")
    return [x == s for s in allowable_set]

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

# ---------- 原子特征提取（78维） ----------
def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(), [
        'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg',
        'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl',
        'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
        'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn',
        'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
        one_of_k_encoding(atom.GetDegree(), list(range(11))) +
        one_of_k_encoding_unk(atom.GetTotalNumHs(), list(range(11))) +
        one_of_k_encoding_unk(atom.GetImplicitValence(), list(range(11))) +
        [atom.GetIsAromatic()])   # 输出维度 78

# ---------- SMILES 转图 ----------
def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError(f"无法解析 SMILES: {smile}")
    canonical_smile = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    mol = Chem.MolFromSmiles(canonical_smile)  # 重新加载规范化分子
    
    c_size = mol.GetNumAtoms()
    features = [atom_features(atom) for atom in mol.GetAtoms()]
    edges = [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in mol.GetBonds()]
    g = nx.Graph(edges).to_directed()
    edge_index = list(g.edges)
    return c_size, features, edge_index

# ---------- 蛋白质序列编码 ----------
seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v: i+1 for i, v in enumerate(seq_voc)}
max_seq_len = 1000

def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict.get(ch, 0)
    return x

# ---------- 主处理函数 ----------
def process_dataset(dataset='davis'):
    print(f'Processing {dataset} dataset...')
    fpath = f'data/{dataset}/'

    train_fold = [e for lst in json.load(open(fpath + "folds/train_fold_setting1.txt")) for e in lst]
    valid_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))
    ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
    affinity = pickle.load(open(fpath + "Y", "rb"), encoding='latin1')

    # SMILES 规范化
    drugs = [Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True, canonical=True) for d in ligands]
    prots = list(proteins.values())

    if dataset == 'davis':
        affinity = -np.log10(affinity / 1e9)

    affinity = np.asarray(affinity)

    # 生成 CSV 文件
    for opt in ['train', 'test']:
        rows, cols = np.where(np.isnan(affinity) == False)
        if opt == 'train':
            rows, cols = rows[train_fold], cols[train_fold]
        else:
            rows, cols = rows[valid_fold], cols[valid_fold]
        with open(f'data/{dataset}_{opt}.csv', 'w') as f:
            f.write('compound_iso_smiles,target_sequence,affinity\n')
            for i in range(len(rows)):
                f.write(f"{drugs[rows[i]]},{prots[cols[i]]},{affinity[rows[i], cols[i]]}\n")

    # 构建图并缓存
    compound_iso_smiles = set(pd.read_csv(f'data/{dataset}_train.csv')['compound_iso_smiles']) | \
                          set(pd.read_csv(f'data/{dataset}_test.csv')['compound_iso_smiles'])
    smile_graph = {smile: smile_to_graph(smile) for smile in compound_iso_smiles}

    # 转换为 PyG 格式并保存
    for opt in ['train', 'test']:
        df = pd.read_csv(f'data/{dataset}_{opt}.csv')
        drugs = np.array(df['compound_iso_smiles'])
        prots = np.array([seq_cat(seq) for seq in df['target_sequence']])
        raw_seq = np.array(df['target_sequence'])
        labels = np.array(df['affinity'])

        processed_file = f'data/processed/{dataset}_{opt}.pt'
        if not os.path.exists(processed_file):
            print(f'Generating {processed_file}...')
            dataset_obj = TestbedDataset(root='data', dataset=f'{dataset}_{opt}',
                                         xd=drugs, xt=prots, y=labels, smile_graph=smile_graph)
        else:
            print(f'{processed_file} already exists.')

# ---------- 执行 ----------
if __name__ == '__main__':
    for ds in ['davis', 'kiba']:
        process_dataset(ds)
