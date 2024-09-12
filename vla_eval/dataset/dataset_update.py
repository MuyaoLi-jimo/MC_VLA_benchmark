from pathlib import Path,PosixPath
from utils import utils
from rich import print
import argparse

DATASET_FOLD = Path(__file__).parent.parent.parent / "data" / "dataset"
DATASET_INDEX_PATH = DATASET_FOLD / "index.json"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="all")
    args = parser.parse_args()
    return args


def update_index():
    """更新类索引 """
    database_fold = DATASET_FOLD
    database_index_path = DATASET_INDEX_PATH
    dataset_index = utils.load_json_file(database_index_path)
    DATASET_LIST = get_dataset_names(dataset_index)
    
    def count_total_num(dataset_tasks:dict):
        total_num = 0
        for num in dataset_tasks.values():
            total_num += num
        return total_num
            
    json_file_paths = database_fold.glob("*.json")
    visited_dataset = set()
    update_flag = False
    for json_file_path in json_file_paths:
        if json_file_path.name == "index.json":
            continue
        dataset,dataset_tasks = get_dataset_task(json_file_path)
        dataset_name = json_file_path.stem
        if dataset_name not in DATASET_LIST:
            DATASET_LIST.add(dataset_name)
            dataset_index[dataset_name] = {
                "id":utils.generate_uuid(),
                "name":dataset_name,
                "timestamp":utils.generate_timestamp(),
                "task":dataset_tasks,
                "attrs":list(list(dataset.values())[0][0].keys()),
                "num":count_total_num(dataset_tasks),
                "available":True
            }
            update_flag = True
        elif dataset_index[dataset_name]["task"] != dataset_tasks:
            dataset_index[dataset_name]["task"] = dataset_tasks
            dataset_index[dataset_name]["timestamp"] = utils.generate_timestamp()
            dataset_index[dataset_name]["num"] = count_total_num(dataset_tasks)
            update_flag = True
        visited_dataset.add(dataset_name)
    for dataset_name in DATASET_LIST:
        if dataset_name not in visited_dataset:
            dataset_index[dataset_name]["available"] = False
            update_flag = True
    DATASET_LIST = visited_dataset
    if update_flag:
        utils.dump_json_file(dataset_index,database_index_path,if_backup=False)
    else:
        print("[red]nothing to change")
                
def get_dataset_names(dataset_index):
    dataset_names = set()
    for dataset_name in dataset_index.keys():
        if dataset_index[dataset_name]["available"]:
            dataset_names.add(dataset_name)
    return dataset_names

def get_dataset_task(json_path:PosixPath):
    json_file = utils.load_json_file(json_path)
    tasks = {}
    for task,qas in json_file.items():
        tasks[task] = len(qas)
    return json_file,tasks

def update_dataset_id(dataset_path):
    dataset_content = utils.load_json_file(dataset_path)
    for key,value in dataset_content.items():
        dataset_content[key] = []
        for entry in value:
            if "id" not in entry:
                entry["id"] = utils.generate_uuid()
            dataset_content[key].append(entry)
    utils.dump_json_file(dataset_content,dataset_path)


def update():
    args = parse_args()
    # 更新index
    update_index()
    if args.name == "all":
        json_file_paths = DATASET_FOLD.glob("*.json")
        for json_file_path in json_file_paths:
            if json_file_path.name == "index.json":
                continue
            update_dataset_id(json_file_path)
    else:
        # 增加id
        dataset_path = DATASET_FOLD / f"{args.name}.json"
        update_dataset_id(dataset_path)
    
def image_update(dataset_name:str):
    dataset_path = DATASET_FOLD / f"{dataset_name}.json"
    dataset = utils.load_json_file(dataset_path)
    for task,lines in dataset.items():
        for idx,_ in enumerate(lines):
            lines[idx]["id"] = utils.generate_uuid()
            lines[idx]["image_base64"] = utils.encode_image_to_base64(lines[idx]["image_path"])
        dataset[task] = lines
    utils.dump_json_file(dataset,dataset_path)
    
def image_no_base64(dataset_name:str):
    dataset_path = DATASET_FOLD / f"{dataset_name}.json"
    dataset = utils.load_json_file(dataset_path)
    for task,lines in dataset.items():
        for idx,_ in enumerate(lines):
            
            del lines[idx]["image_base64"]
        dataset[task] = lines
    utils.dump_json_file(dataset,dataset_path)

if __name__ == "__main__":
    update_dataset_id("/scratch2/limuyao/workspace/VLA_benchmark/data/dataset/visual-advance.json")
