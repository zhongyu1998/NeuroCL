import numpy as np
import os
import random
import time
import torch
import torch.nn as nn

from data import SATDataset
from model import NeuroCL
from numba import njit
from sklearn.metrics import confusion_matrix, accuracy_score
from torch_geometric.loader import DataLoader
from tqdm import tqdm


@njit
def get_neg_vars(num_vars, pos_array):
    mask = np.zeros(num_vars, dtype=np.uint8)
    for i in range(pos_array.size):
        mask[pos_array[i]] = 1

    cnt = 0
    for i in range(num_vars):
        if mask[i] == 0:
            cnt += 1

    neg_array = np.empty(cnt, dtype=np.int32)
    k = 0
    for i in range(num_vars):
        if mask[i] == 0:
            neg_array[k] = i
            k += 1

    return neg_array


def sampling(num_vars, drv_cla_id, drv_var_list, device):
    pos_cla_list, pos_var_list, neg_cla_list, neg_var_list = [], [], [], []

    for cla, pos_vars in zip(drv_cla_id, drv_var_list):
        if len(pos_vars) == 0:
            continue

        all_neg_vars = get_neg_vars(num_vars, np.asarray(pos_vars, dtype=np.int32)).tolist()
        if len(all_neg_vars) > len(pos_vars):
            neg_vars = random.sample(all_neg_vars, len(pos_vars))
        else:
            neg_vars = all_neg_vars

        pos_cla_list.extend([cla] * len(pos_vars))
        pos_var_list.extend(pos_vars)
        neg_cla_list.extend([cla] * len(neg_vars))
        neg_var_list.extend(neg_vars)

    assert len(pos_cla_list) == len(pos_var_list)
    assert len(neg_cla_list) == len(neg_var_list)

    cla_index = torch.tensor(pos_cla_list + neg_cla_list, dtype=torch.long)
    var_index = torch.tensor(pos_var_list + neg_var_list, dtype=torch.long)
    edge_label = torch.cat([torch.ones(len(pos_var_list)), torch.zeros(len(neg_var_list))], dim=0)

    return cla_index.to(device), var_index.to(device), edge_label.to(device)


def train(model, dataset_train, train_loader, optimizer, device, log_file):
    model.train()

    total_loss = 0
    total_var_cnt = 0
    sample_cnt = 0
    all_target = []
    all_pred_class = []

    with tqdm(total=len(dataset_train)) as pbar:
        for data in train_loader:
            if not hasattr(data, "y"):
                log_file.write("No data.y, ignore\n")
                continue
            try:
                data = data.to(device)

                if hasattr(data, "ant_edge_index"):
                    rep = model(data.x, data.edge_index, data.edge_attr, data.ant_edge_index)
                    cla_index, var_index, edge_label = sampling(data.num_vars[0].item(), data.drv_cla_id[0],
                                                                data.drv_var_list[0], device)
                    pred_edge = (rep[cla_index] * rep[var_index]).sum(dim=-1)
                    criterion_edge = nn.BCEWithLogitsLoss()
                    loss_edge = criterion_edge(pred_edge, edge_label)
                else:
                    rep = model(data.x, data.edge_index, data.edge_attr)

                y01_indices = (data.y != -1).nonzero(as_tuple=True)
                y01 = data.y[y01_indices]

                num0 = torch.sum((y01 == 0).int())
                num1 = torch.sum((y01 == 1).int())
                num01 = num0 + num1
                assert num01 == y01.shape[0]

                weight = torch.zeros(num01).to(device)
                weight[y01 == 0] = num01 / (2 * (num0 + 1))
                weight[y01 == 1] = num01 / (2 * (num1 + 1))
                criterion_bb = nn.BCELoss(weight=weight.view(-1, 1))

                pred_pol = model.polarity_prediction(rep)
                pred_bb = pred_pol[y01_indices]
                loss_node = criterion_bb(pred_bb, y01.float().view(-1, 1))

                optimizer.zero_grad(set_to_none=True)

                if hasattr(data, "ant_edge_index"):
                    loss = 0.25 * loss_edge + loss_node
                    loss_edge = loss_edge.item()
                else:
                    loss = loss_node
                loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            except Exception as e:
                if "CUDA out of memory" in str(e):
                    print("CUDA out of memory, skip current data.")
                    continue
                else:
                    raise e

            with torch.no_grad():
                pred_class = (pred_bb >= 0.5).int().flatten()

            total_loss += loss_node.item() * y01.shape[0]
            total_var_cnt += y01.shape[0]

            all_target += y01.cpu().numpy().tolist()
            all_pred_class += pred_class.cpu().numpy().tolist()

            if sample_cnt + hyper_params["batch_size"] <= len(dataset_train):
                pbar.update(hyper_params["batch_size"])
                sample_cnt += hyper_params["batch_size"]
            else:
                pbar.update(len(dataset_train) - sample_cnt)
                sample_cnt = len(dataset_train)

    c = confusion_matrix(all_target, all_pred_class, labels=[0, 1])

    log_file.write("confusion matrix on training set:\n")
    log_file.write(str(c) + "\n")

    if total_var_cnt > 0:
        log_file.write(f"train_loss = {total_loss / total_var_cnt}\n\n")
        return total_loss / total_var_cnt
    else:
        log_file.write(f"total_var_cnt = {total_var_cnt}\n\n")
        return None


def evaluate(model, valid_loader, device, log_file):
    model.eval()

    total_loss = 0
    total_var_cnt = 0
    all_target = []
    all_pred_class = []

    with torch.no_grad():
        for data in tqdm(valid_loader):
            try:
                data = data.to(device)

                y01_indices = (data.y != -1).nonzero(as_tuple=True)
                y01 = data.y[y01_indices]

                num0 = torch.sum((y01 == 0).int())
                num1 = torch.sum((y01 == 1).int())
                num01 = num0 + num1
                assert num01 == y01.shape[0]

                weight = torch.zeros(num01).to(device)
                weight[y01 == 0] = num01 / (2 * (num0 + 1))
                weight[y01 == 1] = num01 / (2 * (num1 + 1))
                criterion = nn.BCELoss(weight=weight.view(-1, 1))

                rep = model(data.x, data.edge_index, data.edge_attr)
            except Exception as e:
                if "CUDA out of memory" in str(e):
                    print("CUDA out of memory, switch to CPU for current data.")
                    model = model.cpu()
                    data = data.cpu()

                    y01_indices = (data.y != -1).nonzero(as_tuple=True)
                    y01 = data.y[y01_indices]

                    num0 = torch.sum((y01 == 0).int()).cpu()
                    num1 = torch.sum((y01 == 1).int()).cpu()
                    num01 = num0 + num1
                    assert num01 == y01.shape[0]

                    weight = torch.zeros(num01).cpu()
                    weight[y01 == 0] = num01 / (2 * (num0 + 1))
                    weight[y01 == 1] = num01 / (2 * (num1 + 1))
                    criterion = nn.BCELoss(weight=weight.view(-1, 1))

                    rep = model(data.x, data.edge_index, data.edge_attr)
                else:
                    raise e

            pred_pol = model.polarity_prediction(rep)
            pred_bb = pred_pol[y01_indices]

            loss = criterion(pred_bb, y01.float().view(-1, 1))
            total_loss += loss.item() * y01.shape[0]
            total_var_cnt += y01.shape[0]

            pred_class = (pred_bb >= 0.5).int().flatten()
            all_target += y01.cpu().numpy().tolist()
            all_pred_class += pred_class.cpu().numpy().tolist()

    c = confusion_matrix(all_target, all_pred_class, labels=[0, 1])

    log_file.write("confusion matrix on validation set:\n")
    log_file.write(str(c) + "\n")

    loss = total_loss / total_var_cnt
    accuracy = accuracy_score(all_target, all_pred_class)

    log_file.write(f"validation_loss = {loss:.4f}\n")
    log_file.write(f"validation_accuracy = {accuracy:.4f}\n")

    return loss, accuracy


if __name__ == '__main__':
    hyper_params = {}

    hyper_params["seed"] = 77
    hyper_params["lr"] = 1e-3
    hyper_params["epoch_num"] = 30
    hyper_params["batch_size"] = 1
    hyper_params["log_dir"] = "./log/train"
    hyper_params["model_dir"] = "./models/train"
    hyper_params["train_dataset_dir"] = "./data/pt/train"
    hyper_params["valid_dataset_dir"] = "./data/pt/validation"
    hyper_params["checkpoint_path"] = ""  # "./models/train/train-best.ptg"

    if not os.path.isdir(hyper_params["log_dir"]):
        os.makedirs(hyper_params["log_dir"])
    if not os.path.isdir(hyper_params["model_dir"]):
        os.makedirs(hyper_params["model_dir"])

    torch.manual_seed(hyper_params["seed"])
    dataset_train = SATDataset(root=hyper_params["train_dataset_dir"])
    dataset_valid = SATDataset(root=hyper_params["valid_dataset_dir"])
    train_loader = DataLoader(dataset_train, batch_size=1, shuffle=True, pin_memory=True, num_workers=2)
    valid_loader = DataLoader(dataset_valid, batch_size=1, shuffle=False, pin_memory=True, num_workers=1)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = NeuroCL(64, 4, 0.25, device).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyper_params["lr"])

    best_acc = 0

    if hyper_params["checkpoint_path"] and os.path.isfile(hyper_params["checkpoint_path"]):
        checkpoint = torch.load(hyper_params["checkpoint_path"], weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        with open(hyper_params["log_dir"] + "/NeuroCL-load.log", "w") as log_file:
            localtime = time.asctime(time.localtime(time.time()))
            log_file.write(str(localtime) + "\n")
            log_file.write("evaluate loaded model on validation set\n\n")

            print("Evaluate loaded model on validation set first.")
            _, acc = evaluate(model, valid_loader, device, log_file)

        best_acc = acc
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"models/train/train-best.ptg")

    for epoch in range(hyper_params["epoch_num"]):
        print(f"---------- epoch: {epoch} ----------\n")

        with open(hyper_params["log_dir"] + f"/NeuroCL-{epoch}.log", "w") as log_file:
            localtime = time.asctime(time.localtime(time.time()))
            log_file.write(str(localtime) + "\n")
            log_file.write(f"epoch: {epoch}\n\n")

            print("train:")
            train_loss = train(model, dataset_train, train_loader, optimizer, device, log_file)
            print()

            print("eval:")
            _, acc = evaluate(model, valid_loader, device, log_file)
            print()

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"models/train/train-{epoch}.ptg")

        if acc > best_acc:
            best_acc = acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, "models/train/train-best.ptg")

    print("Learning completed!")
