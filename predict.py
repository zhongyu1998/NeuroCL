import gzip
import os
import sys
import time
import torch

from tqdm import tqdm
from model import NeuroCL
from sklearn.metrics import accuracy_score


def predict_single(pt_dir, pt_file, model_path, pred_dir, is_cuda=True):
    if pt_file.endswith(".gz"):
        with gzip.open(os.path.join(pt_dir, pt_file), "rb") as f:
            data = torch.load(f, weights_only=False)
    else:
        data = torch.load(os.path.join(pt_dir, pt_file), weights_only=False)

    reverse = data.edge_index.index_select(0, torch.LongTensor([1, 0]))
    data.edge_index = torch.cat([data.edge_index, reverse], dim=1)
    data.edge_attr = torch.cat([data.edge_attr, data.edge_attr], dim=0)

    data.x = data.x.int()
    data.edge_index = data.edge_index.long()

    if is_cuda:
        data = data.cuda()
        model = NeuroCL(64, 4, 0.25, torch.device('cuda'))
        checkpoint = torch.load(model_path, map_location=torch.device('cuda'), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.cuda()
    else:
        data = data.cpu()
        model = NeuroCL(64, 4, 0.25, torch.device('cpu'))
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.cpu()

    model.eval()

    with torch.no_grad():
        rep = model(data.x, data.edge_index, data.edge_attr)
        pred_pol = model.polarity_prediction(rep)
        n2v = data.n2v.cpu().numpy().tolist()

        pred_list = []
        for n, v in enumerate(n2v):
            pred_score = pred_pol[n].tolist()[0]
            pred_list.append((v, pred_score))

        res_list = sorted(pred_list, key=lambda l: l[1], reverse=True)

        cnf_name = pt_file[:-3]
        with open(f"{pred_dir}/{cnf_name}.txt", "w") as f:
            for t in res_list:
                f.write(f"{t[0]},{t[1]}\n")


def predict_cpu(pt_dir, log_dir, model_path, pred_dir):
    pt_file_list = sorted(list(os.listdir(pt_dir)), key=lambda pt_file: os.path.getsize(f"{pt_dir}/{pt_file}"))

    with tqdm(total=len(pt_file_list)) as pbar:
        for pt_file in pt_file_list:
            with open(f"{log_dir}/prediction_time.csv", "a") as f:
                start = time.time()
                try:
                    predict_single(pt_dir, pt_file, model_path, pred_dir, is_cuda=False)
                except Exception as e:
                    print(pt_file, e)
                    break

                time_cost = time.time() - start
                f.write(f"{pt_file},CPU,{time_cost}\n")
                pbar.update()

        print("Prediction completed!")


def predict_gpu(pt_dir, log_dir, model_path, pred_dir):
    pt_file_list = sorted(list(os.listdir(pt_dir)), key=lambda pt_file: os.path.getsize(f"{pt_dir}/{pt_file}"))

    with tqdm(total=len(pt_file_list)) as pbar:
        for pt_file in pt_file_list:
            with open(f"{log_dir}/prediction_time.csv", "a") as f:
                start = time.time()
                try:
                    predict_single(pt_dir, pt_file, model_path, pred_dir, is_cuda=True)
                except Exception as e:
                    print(pt_file, e)
                    break

                time_cost = time.time() - start
                f.write(f"{pt_file},GPU,{time_cost}\n")
                pbar.update()

        print("Prediction completed!")


def predict_mix(pt_dir, log_dir, model_path, pred_dir):
    pt_file_list = sorted(list(os.listdir(pt_dir)), key=lambda pt_file: os.path.getsize(f"{pt_dir}/{pt_file}"))

    processor = "GPU"

    with tqdm(total=len(pt_file_list)) as pbar:
        for pt_file in pt_file_list:
            with open(f"{log_dir}/prediction_time.csv", "a") as f:
                start = time.time()
                if processor == "GPU":
                    try:
                        predict_single(pt_dir, pt_file, model_path, pred_dir, is_cuda=True)
                    except:
                        print("Switch to CPU.")
                        processor = "CPU"
                        start = time.time()
                        predict_single(pt_dir, pt_file, model_path, pred_dir, is_cuda=False)
                else:
                    assert processor == "CPU"
                    predict_single(pt_dir, pt_file, model_path, pred_dir, is_cuda=False)

                time_cost = time.time() - start
                f.write(f"{pt_file},{processor},{time_cost}\n")
                pbar.update()

        print("Prediction completed!")


if __name__ == '__main__':
    processor = "gpu"

    dataset, level = 'k-clique', 'hard'
    mode = f"{dataset}_{level}"

    pt_dir = f"./data/pt/{mode}_test/processed"
    log_dir = f"./log/{mode}"
    model_path = f"./models/{mode}/train-best.ptg"
    pred_dir = f"./prediction/{mode}"

    if not os.path.isdir(pred_dir):
        os.makedirs(pred_dir)

    if processor == "cpu":
        print("Predict backbone on CPU.")
        predict_cpu(pt_dir, log_dir, model_path, pred_dir)
    elif processor == "gpu":
        print("Predict backbone on GPU (with cuda).")
        predict_gpu(pt_dir, log_dir, model_path, pred_dir)
    elif processor == "mix":
        print("Predict backbone on GPU (with cuda) and CPU.")
        predict_mix(pt_dir, log_dir, model_path, pred_dir)
    else:
        print("Error: Processor undefined.")
