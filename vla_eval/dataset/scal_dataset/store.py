from pathlib import Path
import numpy as np
from utils import utils

TEMP_QA_BUFFER = Path("/scratch2/limuyao/workspace/VLA_benchmark/data/dataset/qa_buffer_database")
DATASET_FOLD = Path("/scratch2/limuyao/workspace/VLA_benchmark/data/dataset")

def buffer_qa(q_a,dataset_name,source):
    """将制作好的问题放到一个临时数据集中，等待确认 """
    timestamp = utils.generate_timestamp()
    temp_qa = {
        "timestamp":timestamp,
        "dataset_name":dataset_name,
    }
    if dataset_name in {"visual-advance","visual-basic"}:
        temp_qa["image_path"]= str(source[0])
    temp_qa.update(q_a)
    database = utils.LmdbProcessor(TEMP_QA_BUFFER)
    database.insert(key=q_a["id"],value=temp_qa)
    
def examine_qa():
    """从数据库中提取一个预备问题 """
    database = utils.LmdbProcessor(TEMP_QA_BUFFER)
    key = np.random.choice(database.get_all_keys())
    line = database.get(key)
    return line["timestamp"],line["id"],line["dataset_name"],line.get("image_path",""),line["question"],line["answer"],line.get("explanation","")

def confirm_qa(confirm_flag,dataset_name,task_name,timestamp,uuid,question,answer,explanation,path):
    """如果确认保留这个数据集，将数据加入到对应的数据集中，然后在index中标记  """
    database = utils.LmdbProcessor(TEMP_QA_BUFFER)
    delete_flag = database.delete(uuid)
    if confirm_flag and delete_flag:
        q_a = {
            "id":uuid,
            "question":question,
            "answer":answer,
        }
        if path:
            q_a["image_path"] = path
        if explanation:
            q_a["explanation"] = explanation
        # 加入测试数据集
        dataset_path = DATASET_FOLD / f"{dataset_name}.json"
        dataset_file = utils.load_json_file(dataset_path)
        dataset_file[task_name].append(q_a)
        utils.dump_json_file(dataset_file,dataset_path)
        
        index_path = DATASET_FOLD / "index.json"
        index_file = utils.load_json_file(index_path)
        index_file[dataset_name]["update"] = False
        utils.dump_json_file(index_file,index_path)
        
        