import torch
from torch_geometric.data import Data
import pandas as pd
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from sklearn.metrics import precision_score, recall_score, f1_score
from model import GCN
import os
import sys
import clip

# 指定不同的下载源或手动下载
try:
    # 尝试不同的下载根目录
    model_path = clip._download(clip._MODELS["ViT-B/32"], download_root="./clip_models")
    clip_model, preprocess = clip.load(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
except:
    # 如果仍然失败，尝试使用默认方式
    clip_model, preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu", jit=False)

# 添加当前目录到路径，确保可以导入预处理模块
sys.path.append(os.path.dirname(__file__))

try:
    from preprocess import preprocess_dataset, check_dataset_structure
except ImportError:
    print("警告: 无法导入预处理模块")


def ensure_datasets_ready():
    """确保所有数据集都已准备就绪"""
    datasets = ['pheme', 'twitter', 'weibo']

    for dataset in datasets:
        gcn_train_path = f'dataset/{dataset}/dataforGCN_train.csv'

        if not os.path.exists(gcn_train_path):
            print(f"检测到缺少 {gcn_train_path}，执行预处理...")
            try:
                # 检查数据集结构
                check_dataset_structure(dataset)
                # 执行预处理
                success = preprocess_dataset(dataset)
                if not success:
                    print(f"错误: {dataset} 数据集预处理失败")
                    return False
            except Exception as e:
                print(f"预处理过程中出错: {e}")
                return False
        else:
            print(f"{dataset} 数据集已就绪")

    return True


# 在主程序开始前调用
if __name__ == "__main__":
    if not ensure_datasets_ready():
        print("数据集准备失败，程序退出")
        sys.exit(1)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

setup_seed(0)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="pheme")
args = parser.parse_args()

dataset_name = args.dataset_name

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data = pd.read_csv('dataset/' + dataset_name + '/dataforGCN_train.csv')
test_data = pd.read_csv('dataset/' + dataset_name + '/dataforGCN_test.csv')

tweet_embeds = torch.load('dataset/' +dataset_name+ '/TweetEmbeds.pt', map_location=device)
tweet_graph = torch.load('dataset/' + dataset_name + '/TweetGraph.pt', map_location=device)

pseudo_path = f'dataset/{dataset_name}/{dataset_name}_analysis_results.csv'
if not os.path.exists(pseudo_path):
    raise FileNotFoundError(f"缺少伪标签文件: {pseudo_path}")

psesudo_data = pd.read_csv(pseudo_path)

psesudo_labels = torch.tensor(psesudo_data["analysis"].tolist(), dtype=torch.long).to(device)
psesudo_probs = torch.tensor(psesudo_data["prob"].tolist(), dtype=torch.float).to(device)

# --- FIX: Twitterillä labelit ovat stringejä, muunnetaan numeroksi ---
def normalize_labels(labels):
    label_map = {
        "fake": 1, "real": 0,
        "false": 1, "true": 0,
        "0": 0, "1": 1
    }

    normalized = []
    for x in labels:
        s = str(x).lower().strip()
        if s in label_map:
            normalized.append(label_map[s])
        else:
            # fallback: jos tuntematon → 0
            normalized.append(0)
    return normalized

label_list_train = normalize_labels(train_data["label"].tolist())
label_list_test  = normalize_labels(test_data["label"].tolist())

labels_train_tensor = torch.tensor(label_list_train, dtype=torch.long)
labels_test_tensor  = torch.tensor(label_list_test, dtype=torch.long)

labels = torch.cat([labels_train_tensor, labels_test_tensor], dim=0)


data = Data(
    x=tweet_embeds.float(),
    edge_index=tweet_graph.coalesce().indices(),
    edge_attr=None,
    train_mask=torch.tensor([True]*len(label_list_train) + [False]*(len(labels)-len(label_list_train))).bool(),
    test_mask=torch.tensor([False]*len(label_list_train) + [True]*(len(labels)-len(label_list_train))).bool(),
    y=labels
).to(device)
num_features = tweet_embeds.shape[1]
num_classes = 2

data.x = torch.cat([data.x, torch.zeros((data.num_nodes, num_classes), device=device)], dim=1)

class UniMP(torch.nn.Module):
    def __init__(self, in_channels, num_classes, hidden_channels, num_layers,
                heads, dropout=0.3):
        super().__init__()

        self.num_classes = num_classes

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            if i < num_layers:
                out_channels = hidden_channels // heads
                concat = True
            else:
                out_channels = num_classes
                concat = False
            conv = TransformerConv(in_channels, out_channels, heads,
                                concat=concat, beta=True, dropout=dropout)
            self.convs.append(conv)
            in_channels = hidden_channels

            if i < num_layers:
                self.norms.append(torch.nn.LayerNorm(hidden_channels))

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs[:-1], self.norms):
            x = norm(conv(x, edge_index)).relu()
        x = self.convs[-1](x, edge_index)
        return x

data.y = data.y.view(-1)
model = GCN(
    in_feature=num_features + num_classes,
    hidden=64,
    out_feature=num_classes,
    dropout=0.3
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)

train_mask = data.train_mask
test_mask = data.test_mask
test_mask_idx = data.test_mask.nonzero(as_tuple=False).view(-1)

def train(label_rate=0.95):
    model.train()

    data.x[:, -num_classes:] = 0

    train_mask_idx = train_mask.nonzero(as_tuple=False).view(-1)
    mask = torch.rand(train_mask_idx.shape[0]) < label_rate
    train_labels_idx = train_mask_idx[mask]  
    train_unlabeled_idx = train_mask_idx[~mask] 

    
    num_pseudo = int(len(test_mask_idx) * 0.05)
    topk_indices = torch.topk(psesudo_probs, num_pseudo).indices

    test_psesudo_idx = test_mask_idx[topk_indices]
    selected_psesudo_labels = psesudo_labels[topk_indices] 
    data.x[
        torch.cat([train_labels_idx, test_psesudo_idx]), 
        -num_classes:
    ] = F.one_hot(
        torch.cat([data.y[train_labels_idx], selected_psesudo_labels]), 
        num_classes
    ).float()

    optimizer.zero_grad()
    out = model(data)

    loss = F.cross_entropy(out[train_unlabeled_idx], data.y[train_unlabeled_idx])
    loss.backward()
    optimizer.step()

    use_labels = True
    n_label_iters = 1

    if use_labels and n_label_iters > 0:
        unlabel_idx = torch.cat([train_unlabeled_idx, data.test_mask.nonzero(as_tuple=False).view(-1)])
        with torch.no_grad():
            for _ in range(n_label_iters):
                torch.cuda.empty_cache()
                out = out.detach()
                data.x[unlabel_idx, -num_classes:] = F.softmax(out[unlabel_idx], dim=-1)
                out = model(data)

    return loss.item()

max_test_acc = 0
max_precision = 0
max_recall = 0
max_f1 = 0

best_epoch = 0

@torch.no_grad()
def test():
    model.eval()

    data.x[:, -num_classes:] = 0

    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    test_idx = data.test_mask.nonzero(as_tuple=False).view(-1)
    
    data.x[train_idx, -num_classes:] = F.one_hot(data.y[train_idx], num_classes).float()

    unlabel_idx = data.test_mask.nonzero(as_tuple=False).view(-1)
    n_label_iters = 1
    for _ in range(n_label_iters):
        out = model(data)

        data.x[unlabel_idx, -num_classes:] = F.softmax(out[unlabel_idx], dim=-1)

    out = model(data)
    pred = out[test_mask].argmax(dim=-1)
    
    y_true = data.y[test_mask].cpu().numpy()
    y_pred = pred.cpu().numpy()

    test_acc = (pred == data.y[test_mask]).sum().item() / pred.size(0)

    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    val_acc = 0

    return val_acc, test_acc, precision, recall, f1

best_precision = 0
best_recall = 0
best_f1 = 0

for epoch in range(1, 3001):
    loss = train()
    val_acc, test_acc, precision, recall, f1 = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}, '
    f'Test Acc: {test_acc:.4f}, Precision: {precision:.4f}, '
    f'Recall: {recall:.4f}, F1: {f1:.4f}')
    
    if test_acc > max_test_acc:
        max_test_acc = test_acc
        best_precision = precision
        best_recall = recall
        best_f1 = f1
        best_epoch = epoch 

print(f'Best Epoch: {best_epoch}, Max Test Acc: {max_test_acc:.4f}, '
    f'Precision: {best_precision:.4f}, Recall: {best_recall:.4f}, F1: {best_f1:.4f}')
