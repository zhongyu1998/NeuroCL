import os
import shutil
import subprocess
import threading

from multiprocessing import Pool
from pysat.examples.rc2 import RC2
from pysat.formula import CNF, WCNF
from tqdm import tqdm


def gen_bb(n_cpu=1):
    n_instances = len([file_name for file_name in os.listdir(src_cnf_dir) if file_name.endswith(".cnf")])

    with Pool(n_cpu) as p:
        with tqdm(total=n_instances) as pbar:
            for _ in p.imap_unordered(gen_bb_single, range(n_instances)):
                pbar.update()


def gen_bb_single(i):
    src_cnf_file = src_cnf_dir + f"/{i:05}.cnf"
    tgt_cnf_file = tgt_cnf_dir + f"/{level}_{dataset}_{label}_{i:05}.cnf"
    shutil.copy2(src_cnf_file, tgt_cnf_file)
    cnf = CNF(from_file=src_cnf_file)

    if label == "unsat":
        wcnf = WCNF()
        for clause in cnf.clauses:
            wcnf.append(clause, weight=1)

        with RC2(wcnf) as rc2:
            timer = threading.Timer(5000, timedout_interrupt, args=[rc2])
            timer.start()
            model = rc2.compute(expect_interrupt=True)
            timer.cancel()
            rc2.clear_interrupt()

        if model is None:
            with open(fail_file_path, "a") as fail_file:
                fail_file.write(tgt_cnf_file.split("/")[-1] + "\n")
            os.remove(tgt_cnf_file)
            return

        maxsat_clauses = []
        for clause in cnf.clauses:
            if any(lit in model for lit in clause):
                maxsat_clauses.append(clause)

        if maxsat_clauses:
            max_var = max(abs(lit) for msc in maxsat_clauses for lit in msc)
        else:
            max_var = 0

        out_cnf = CNF()
        out_cnf.nv = max_var
        for msc in maxsat_clauses:
            out_cnf.append(msc)
        tmp_cnf_file = tgt_cnf_dir + f"/tmp_{level}_{dataset}_{label}_{i:05}.cnf"
        out_cnf.to_file(tmp_cnf_file)

    cnf_file_path = tmp_cnf_file if label == "unsat" else tgt_cnf_file
    backbone_file_path = backbone_dir + f"/{level}_{dataset}_{label}_{i:05}.cnf.backbone"
    cmd = ["./cadiback/cadiback", cnf_file_path, backbone_file_path]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 10

    with open(backbone_file_path, "r") as bb_file:
        line = bb_file.readline().strip()
        while len(line) == 0:
            line = bb_file.readline().strip()
        lit = int(line.split()[-1])
        if lit == 0:
            with open(fail_file_path, "a") as fail_file:
                fail_file.write(tgt_cnf_file.split("/")[-1] + "\n")
            os.remove(tgt_cnf_file)
            os.remove(backbone_file_path)

    if label == "unsat":
        os.remove(tmp_cnf_file)


def timedout_interrupt(r):
    r.interrupt()


if __name__ == '__main__':
    dataset, level, stage, label = 'k-clique', 'hard', 'test', 'unsat'

    src_cnf_dir = f"/mnt/shared-storage-user/xxx/G4SATBench/data/{level}/{dataset}/{stage}/{label}"
    tgt_cnf_dir = f"./data/cnf/{dataset}_{level}_{stage}"
    backbone_dir = f"./data/backbone/{dataset}_{level}_{stage}"
    fail_file_path = tgt_cnf_dir + "/fail_files.txt"

    if not os.path.isdir(tgt_cnf_dir):
        os.makedirs(tgt_cnf_dir)
    if not os.path.isdir(backbone_dir):
        os.makedirs(backbone_dir)

    gen_bb(n_cpu=8)

    if os.path.isfile(fail_file_path):
        print("The following CNF files failed to generate backbone:")
        with open(fail_file_path, "r") as fail_file:
            content = fail_file.read()
            print(content)
    else:
        print("Successfully generated backbone for all CNF files.")
