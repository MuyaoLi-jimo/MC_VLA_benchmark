import numpy as np
import random
from rich import print
from pathlib import Path,PosixPath
from utils import utils
from vla_eval.dataset import dataset_wrapper
import pandas as pd

TEXT_FOLD  = Path("/nfs-shared/pretrain-jarvis-data/retrieve/data/text/processed_text") #下面是网站文件夹，再往下是大量markdown文档
IMAGE_FOLD = Path("/scratch2/limuyao/muyao/workspace/jarvis/1stage/out/keyframes/image/7xx") 
EXAMPLE_FOLD = Path("/scratch2/limuyao/workspace/VLA_benchmark/data/dataset")
INDEX_FOLD = EXAMPLE_FOLD / "index.json"

def get_tasks_map():
    dataset = utils.load_json_file(INDEX_FOLD)
    dataset_task_map = {}
    for dataset_names in dataset:
        dataset_task_map[dataset_names] = dataset[dataset_names]["task"]
    return dataset_task_map

def get_dataset_num():
    dataset = utils.load_json_file(INDEX_FOLD)
    dataset_nums = {"total":0}
    for dataset_names in dataset:
        dataset_nums[dataset_names] = dataset[dataset_names]["num"]
        dataset_nums["total"]+=dataset[dataset_names]["num"]
    df = pd.DataFrame({
        'Dataset Name': list(dataset_nums.keys()),
        'Number': list(dataset_nums.values())
    })
    return df

def get_source_data(dataset_name:str="knowledge"):
    source_data = []
    if dataset_name in {"knowledge","reason"}:
        child_folds = [child for child in TEXT_FOLD.iterdir() if child.is_dir()]
        child_fold = np.random.choice(child_folds)
        text_paths = list(child_fold.iterdir())
        text_path = random.sample(text_paths,1)[0]
        source_data = [text_path]
    elif dataset_name in {"visual-advance","visual-basic"}:
        child_folds = [child for child in IMAGE_FOLD.iterdir() if child.is_dir()]
        child_fold = np.random.choice(child_folds)
        image_paths = list(child_fold.iterdir())
        image_path = random.sample(image_paths,1)[0]
        source_data = [image_path]
    else:
        raise Exception(f"Unsupported Type {dataset_name}")
    return source_data

def get_example(dataset_name:str="knowledge",task=""):
    dataset = dataset_wrapper.make(dataset_name)
    example = dataset.sample(task)
    return example



if __name__ == "__main__":
    print(get_dataset_num())