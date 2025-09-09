import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import MessagePassing

class GINConvNetWithCurvature(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=78, num_features_xt=25,
                 embed_dim=128, output_dim=128, dropout=0.2, lstm_hidden_dim=256, lstm_layers=1):
        super(GINConvNetWithCurvature, self).__init__()

        dim = 32  # 保持各层维度一致
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output

        # 特征转换层
        self.feature_transform = Linear(num_features_xd, dim)

        # 自定义GINConv层定义（保持不变）
        class GINConvWithCurvature(MessagePassing):
            def __init__(self, nn_model, eps=0, train_eps=False):
                super(GINConvWithCurvature, self).__init__(aggr='add')
                self.nn = nn_model
                self.eps = torch.nn.Parameter(torch.Tensor([eps])) if train_eps else eps

            def forward(self, x, edge_index):
                edge_weight = self.compute_curvature(edge_index, x).to(x.device)
                out = (1 + self.eps) * x
                out = out + self.propagate(edge_index, x=x, edge_weight=edge_weight)
                return self.nn(out)

            def message(self, x_j, edge_weight):
                return edge_weight.view(-1, 1) * x_j if edge_weight is not None else x_j

            def compute_curvature(self, edge_index, x):
                row, col = edge_index
                deg = torch.bincount(edge_index.flatten(), minlength=x.size(0))
                return 4 - (deg[row] + deg[col])

        # 创建GIN层（保持输入输出维度相同）
        def create_gin_layer():
            return Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))

        # 定义5个GIN层 + BatchNorm
        self.conv1 = GINConvWithCurvature(create_gin_layer())
        self.bn1 = nn.BatchNorm1d(dim)
        
        self.conv2 = GINConvWithCurvature(create_gin_layer())
        self.bn2 = nn.BatchNorm1d(dim)
        
        self.conv3 = GINConvWithCurvature(create_gin_layer())
        self.bn3 = nn.BatchNorm1d(dim)
        
        self.conv4 = GINConvWithCurvature(create_gin_layer())
        self.bn4 = nn.BatchNorm1d(dim)
        
        self.conv5 = GINConvWithCurvature(create_gin_layer())
        self.bn5 = nn.BatchNorm1d(dim)

        # 后续全连接层
        self.fc1_xd = Linear(dim, output_dim)

        # LSTM部分（保持不变）
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, lstm_hidden_dim, lstm_layers,
            dropout=dropout, batch_first=True, bidirectional=True
        )
        self.attn = nn.Linear(lstm_hidden_dim * 2, 1)
        self.fc1_xt = nn.Linear(lstm_hidden_dim * 2, output_dim)

        # 融合与输出层
        self.fc1 = nn.Linear(output_dim * 2, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, n_output)

    def forward(self, data):
        # GIN部分前向传播
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.feature_transform(x)

        # 残差连接实现（核心修改部分）
        x1 = self.relu(self.conv1(x, edge_index))  # 第1层无残差
        x1 = self.bn1(x1)
        
        x2 = self.relu(self.conv2(x1, edge_index)) + x1  # 残差连接
        x2 = self.bn2(x2)
        
        x3 = self.relu(self.conv3(x2, edge_index)) + x2  # 残差连接
        x3 = self.bn3(x3)
        
        x4 = self.relu(self.conv4(x3, edge_index)) + x3  # 残差连接
        x4 = self.bn4(x4)
        
        x5 = self.relu(self.conv5(x4, edge_index)) + x4  # 残差连接
        x5 = self.bn5(x5)

        # 全局池化
        x = global_add_pool(x5, batch)
        x = self.relu(self.fc1_xd(x))
        x = self.dropout(x)

        # LSTM部分
        target = data.target
        embedded_xt = self.embedding_xt(target)
        lstm_out, _ = self.lstm(embedded_xt)
        attn_weights = F.softmax(self.attn(lstm_out), dim=1)
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)
        xt = self.relu(self.fc1_xt(context_vector))

        # 特征融合
        xc = torch.cat([x, xt], dim=1)
        xc = self.relu(self.fc1(xc))
        xc = self.dropout(xc)
        xc = self.relu(self.fc2(xc))
        xc = self.dropout(xc)
        
        return self.out(xc)
