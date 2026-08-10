import gzip
import os
import subprocess
import sys
import torch

from multiprocessing import Pool
from torch_geometric.data import Data
from tqdm import tqdm
from xtract import xtract


def gen_pt(cnf_dir, trace_dir, pt_dir, n_cpu=1):
    task_list = []
    for file_name in sorted(os.listdir(cnf_dir)):
        file_path = cnf_dir + "/" + file_name
        if os.path.isfile(file_path):
            if (file_name.endswith(".xz") or file_name.endswith(".bz2") or
                file_name.endswith(".lzma") or file_name.endswith(".gz")) and \
                not os.path.isfile(pt_dir + "/" + ".".join(file_name.split(".")[:-1]) + ".pt"):
                task_list.append([cnf_dir, file_name, trace_dir, pt_dir])

    with Pool(n_cpu) as p:
        with tqdm(total=len(task_list)) as pbar:
            for _ in p.imap_unordered(gen_pt_single, task_list):
                pbar.update()

    print("Parallel generation finished.")


def gen_pt_single(arg_list):
    cnf_dir, cnf_name, trace_dir, pt_dir = arg_list[0], arg_list[1], arg_list[2], arg_list[3]

    cnf_file_path = cnf_dir + "/" + cnf_name
    if cnf_name.endswith(".xz"):
        dc_cnf_path = cnf_file_path[0:-3]
    elif cnf_name.endswith(".gz"):
        dc_cnf_path = cnf_file_path[0:-3]
    elif cnf_name.endswith(".lzma"):
        dc_cnf_path = cnf_file_path[0:-5]
    elif cnf_name.endswith(".bz2"):
        dc_cnf_path = cnf_file_path[0:-4]
    else:
        print(f"Warning: Unknown compress format: {cnf_name}.")
        return

    if os.path.exists(dc_cnf_path):
        os.remove(dc_cnf_path)
    xtract(cnf_file_path, cnf_dir)
    assert os.path.isfile(dc_cnf_path)

    cnf_name = ".".join(cnf_name.split(".")[:-1])
    backbone_name = cnf_name + ".backbone.xz"
    backbone_dir = "./data/backbone/" + cnf_dir.split("/")[-1]
    backbone_file_path = backbone_dir + "/" + backbone_name

    if os.path.isfile(backbone_file_path):
        dc_backbone_path = backbone_file_path[:-3]
        if os.path.exists(dc_backbone_path):
            os.remove(dc_backbone_path)
        xtract(backbone_file_path, backbone_dir)
    else:
        print(f"Warning: Backbone file does not exist: {backbone_file_path}.")
        dc_backbone_path = None

    trace_name = cnf_name + ".lrat"
    trace_file_path = trace_dir + "/" + trace_name

    data = cnf_to_bipartite(dc_cnf_path, dc_backbone_path, trace_file_path)

    os.remove(dc_cnf_path)
    if os.path.isfile(dc_backbone_path):
        os.remove(dc_backbone_path)
    if os.path.isfile(trace_file_path):
        os.remove(trace_file_path)

    if data is None:
        return

    pt_file_path = pt_dir + "/" + cnf_name + ".pt"
    try:
        if os.path.isfile(pt_file_path):
            os.remove(pt_file_path)
        torch.save(data, pt_file_path)
    except Exception as e:
        print(e)
        tmp_dir = "/".join(pt_dir.split("/")[:-1]) + "/tmp"
        if not os.path.isdir(tmp_dir):
            os.makedirs(tmp_dir)
        tmp_file_path = tmp_dir + "/" + cnf_name + ".pt"
        torch.save(data, tmp_file_path)
        print(f"Temporarily save to {tmp_file_path}.")


def cnf_to_bipartite(cnf_file_path, backbone_file_path, trace_file_path):
    backbone = set()
    if backbone_file_path is not None:
        with open(backbone_file_path, "r") as f:
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    lit = int(line.split()[-1])
                    if lit != 0:
                        backbone.add(lit)

        if len(backbone) == 0:
            print(f"Warning: No backbone in the file: {backbone_file_path}.")
            return None

    X = []
    v2n = {}
    num_vars = 0
    with open(cnf_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue

            if line[0] == "c" or line[0] == "p":
                continue
            else:
                lit_list = [int(lit) for lit in line.split()[:-1]]
                for lit in lit_list:
                    var = abs(lit)
                    if var not in v2n:
                        v2n[var] = len(X)
                        X.append(0)
                        num_vars += 1

    y = []
    if backbone_file_path is not None:
        y = [-1 for _ in range(num_vars)]
        for var, node_id in v2n.items():
            if var in backbone:
                assert -var not in backbone
                y[node_id] = 1
            elif -var in backbone:
                y[node_id] = 0
        assert len(X) == len(y)
    assert len(y) > 0 and set(y) != {-1}

    edge_index = []
    edge_attr = []
    with open(cnf_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0 or line[0] == "c" or line[0] == "p":
                continue
            else:
                lit_list = [int(lit) for lit in line.split()[:-1]]
                cla_node_id = len(X)
                X.append(1)

                for lit in lit_list:
                    var = abs(lit)
                    var_node_id = v2n[var]
                    edge_index.append([var_node_id, cla_node_id])
                    if lit > 0:
                        edge_attr.append(1)
                    else:
                        assert lit < 0
                        edge_attr.append(-1)

    assert len(edge_index) == len(edge_attr)

    max_conflicts = 10
    ant_edge_index = []
    drv_cla_id = []
    drv_var_list = []
    if training_flag:
        cmd = ["./cadical/build/cadical", "-c", str(max_conflicts),
               "--quiet", "--no-binary", "--lrat", cnf_file_path, trace_file_path]
        cnf_name = cnf_file_path.split("/")[-1]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except Exception as e:
            print(f"Failed with: {e} for {cnf_name}.")

        with open(trace_file_path, "r") as f:
            for line in f:
                if len(X) > 10000:
                    break

                line = line.strip()
                if len(line) == 0 or "d" in line:
                    continue
                else:
                    num_list = [int(num) for num in line.split()[:-1]]
                    cla_node_id = len(X)
                    assert cla_node_id + 1 == num_list[0] + num_vars
                    X.append(-1)

                    index = num_list.index(0)

                    var_list = []
                    for lit in num_list[1:index]:
                        var = abs(lit)
                        var_node_id = v2n[var]
                        var_list.append(var_node_id)
                    drv_cla_id.append(cla_node_id)
                    drv_var_list.append(var_list)

                    for hint in num_list[index+1:]:
                        ant_edge_index.append([hint+num_vars-1, cla_node_id])

    X = torch.tensor(X, dtype=torch.int8)
    edge_index = torch.tensor(edge_index, dtype=torch.int32)
    edge_attr = torch.tensor(edge_attr, dtype=torch.int8)

    n2v = [-1 for _ in range(len(v2n))]
    for v, n in v2n.items():
        n2v[n] = v
    assert all(i != -1 for i in n2v)
    n2v = torch.tensor(n2v, dtype=torch.int32)

    if len(ant_edge_index) > 0:
        ant_edge_index = torch.tensor(ant_edge_index, dtype=torch.int32)
        if len(y) > 0:
            y = torch.tensor(y, dtype=torch.int8)
            data = Data(x=X, n2v=n2v, y=y, num_vars=num_vars, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr,
                        ant_edge_index=ant_edge_index.t().contiguous(), drv_cla_id=drv_cla_id, drv_var_list=drv_var_list)
        else:
            data = Data(x=X, n2v=n2v, num_vars=num_vars, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr,
                        ant_edge_index=ant_edge_index.t().contiguous(), drv_cla_id=drv_cla_id, drv_var_list=drv_var_list)
    else:
        if len(y) > 0:
            y = torch.tensor(y, dtype=torch.int8)
            data = Data(x=X, n2v=n2v, y=y, num_vars=num_vars, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr)
        else:
            data = Data(x=X, n2v=n2v, num_vars=num_vars, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr)

    return data


if __name__ == '__main__':
    mode = sys.argv[1]

    if mode == "train":
        training_flag = True
    elif mode == "valid" or mode == "test":
        training_flag = False
    else:
        print("Error: Command-line argument can only be train, valid, or test.")
        sys.exit(1)

    cnf_dir = "./data/cnf/" + mode

    if os.path.isdir(cnf_dir):
        trace_dir = "./data/trace/" + mode
        if not os.path.isdir(trace_dir):
            os.makedirs(trace_dir)
        pt_dir = "./data/pt/" + mode + "/processed"
        if not os.path.isdir(pt_dir):
            os.makedirs(pt_dir)
            os.makedirs("./data/pt/" + mode + "/raw")
        gen_pt(cnf_dir, trace_dir, pt_dir, n_cpu=8)
    else:
        print("Error: CNF directory name is missing! Please rerun this program with the directory name.")
