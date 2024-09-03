"""注册新模型
    1. 先测试，测试通过才放入文档中
    2. 添加各种属性
"""

import argparse
import time
from pathlib import Path
from rich import print
from vla_eval.model import model
from utils import utils

MODEL_FOLD = Path("/nfs-shared/models")
MODEL_ATTR_FOLD = Path(__file__).parent.parent.parent / "data" / "model"
MODEL_INDEX_PATH = MODEL_ATTR_FOLD / "model.json"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="10003")
    parser.add_argument("--model_path", type=str, default="") #默认是MODEL_FOLD/model_name
    parser.add_argument("--model_type", type=str, default="temp")
    parser.add_argument("--support_vision", type=bool, default=True)
    args = parser.parse_args()
    return args

def insert_model(model_name,model_path,model_type,support_vision,model_port,timestamp):
    
    model_index = utils.load_json_file(MODEL_INDEX_PATH)
    if model_name in model_index:
        print(f"[red]你确定要重置{model_name}吗？")
        raise AssertionError
    
    log_path = MODEL_ATTR_FOLD / "log" / f"{model_name}.log"
    
    # 验证
    if model_type != "temp":
        pid,_ = model.run_vllm_server(devices=[],device_num=1,model_path=model_path,log_path=log_path,port=9009,max_model_len=2048,gpu_memory_utilization=0.95)
        model.stop_vllm_server(pid)
    
    # 填充信息：
    model_attr = {
        "id":utils.generate_uuid(),
        "insert_time":timestamp,
        "path":model_path,
        "avaliable":True,
        "type":model_type,
        "support vision":support_vision,
        "elo rating":{"total":{"mu":model.MY_MU,
                               "sigma":model.MY_SIGMA,}},
        "task score":{},
        "running":False,
    }
    if model_type != "commercial":
        model_attr["pid"] = 0
        model_attr["port"] = 0
        model_attr["host"] = ""
    if model_type == "temp":
        model_attr["running"] =True
        model_attr["pid"] = 0
        model_attr["port"] = model_port
        model_attr["host"] = "localhost"
    
    # 存入
    model_index = utils.load_json_file(MODEL_INDEX_PATH)
    model_index[model_name] = model_attr
    utils.dump_json_file(model_index,MODEL_INDEX_PATH)
    print(f"[bold blue]已完成 {model_name} 的注册")

if __name__ == "__main__":
    args = parse_args() #得到参数
    timestamp = utils.generate_timestamp()
    model_name = args.model_name
    model_path = str(args.model_path) if args.model_path != "" else str(MODEL_FOLD / model_name)
    model_type = args.model_type
    model_port = None
    if model_type == "temp":
        model_name = model_name + "-" + f"{timestamp}"
        model_path = ""
        model_port = int(model_name)
    support_vision = args.support_vision
    insert_model(model_name,model_path,model_type,support_vision,model_port,timestamp)