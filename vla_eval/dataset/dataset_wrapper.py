""" 
获取数据集的api
"""

from pathlib import Path,PosixPath
from utils import utils
from vla_eval.dataset.visual_base import VisualBaseDataset
from vla_eval.dataset.text_base import TextbaseDataset


DATASET_FOLD = Path(__file__).parent.parent.parent / "data" / "dataset"
DATASET_INDEX_PATH = DATASET_FOLD / "index.json"

def get_avaliable_dataset_names(dataset_index:dict={}):
    if dataset_index=={}: 
        dataset_index = utils.load_json_file(DATASET_INDEX_PATH)
    dataset_names = set(dataset_index.keys())
    return dataset_names
    
def get_dataset_type_map(dataset_index:dict={}):
    if dataset_index=={}: 
        dataset_index = utils.load_json_file(DATASET_INDEX_PATH)
    dataset_map = {}
    for dataset_name,value in dataset_index.items():
        dataset_map[dataset_name] = value["type"]
    return dataset_map

def make(dataset_name:str):
    """得到该验证集的api """
    dataset_map = get_dataset_type_map()
    try:
        dataset_type = dataset_map[dataset_name]
    except KeyError as ke:
        print(f"invalid dataset: {dataset_name}, {ke}")
        exit()
    if dataset_type == "text":
        dataset = TextbaseDataset(dataset_name)
    elif dataset_type == "visual":
        dataset= VisualBaseDataset(dataset_name)
    else:
        raise AssertionError
    return dataset

