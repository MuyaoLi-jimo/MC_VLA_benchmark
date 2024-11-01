"""注册新模型
    1. 先测试，测试通过才放入文档中
    2. 添加各种属性
"""

import argparse
import time
from pathlib import Path
from rich import print
from vla_eval.model import model,inference_model
from utils import utils

MODEL_FOLD = Path("/nfs-shared/models")
MODEL_ATTR_FOLD = Path(__file__).parent.parent.parent / "data" / "model"
MODEL_INDEX_PATH = MODEL_ATTR_FOLD / "model.json"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="molmo-72b-0924")
    parser.add_argument("--model_path", type=str, default="") #默认是MODEL_FOLD/model_name
    parser.add_argument("--model_type", type=str, default="temp")  #pretrained
    parser.add_argument("--support_vision", type=bool, default=True)
    parser.add_argument("--chat_template",type=str, default="")
    parser.add_argument("--model_base",type=str,default="")
    parser.add_argument("--port","-p", type=int, default=0)  #pretrained
    parser.add_argument("--host","-i", type=str, default="localhost")  #pretrained
    args = parser.parse_args()
    return args

def insert_model(model_name,model_path,model_type,support_vision,model_port,timestamp,log_path,chat_template=None,model_base=None,model_host="localhost"):
    model_index = utils.load_json_file(MODEL_INDEX_PATH)
    if model_name in model_index:
        if model_name!="gpt-4o-mini":
            error_message = f"你确定要重置{model_name}吗？"
            print(f"[red]{error_message}")
            return False,error_message
        else:
            try:
                inference_model.inference_model(model_name) #内部已经停止
                return True,"regenerate"
            except Exception as e:
                return False,e
            
    # 验证
    if model_type not in { "temp" ,"commercial"}:
        try: 
            pid,_ = model.run_vllm_server(devices=[],device_num=1,model_path=model_path,log_path=log_path,port=9009,max_model_len=2048,gpu_memory_utilization=0.95)
        except Exception as e:
            print(e)
            return False,e
    
    # 填充信息：
    model_attr = {
        "id":utils.generate_uuid(),
        "insert_time":timestamp,
        "path":model_path,
        "avaliable":True, #是否可以测评
        "runable":True,   #是否可以online推理
        "type":model_type,
        "support vision":support_vision,
        "OE done":[],
        "model elo rating":{"total":{"mu":model.MY_MU,
                               "sigma":model.MY_SIGMA,
                               "win":0,}},
        "human elo rating":{"total":{"mu":model.MY_MU,
                               "sigma":model.MY_SIGMA,
                               "win":0,}},
        "MCQ score":{},
        "running state":{}
    }
    if chat_template!="":
        model_attr["template"] = chat_template
    if model_base:
        model_attr["base"] = model_base
    model_attr["running state"]["running"]=True
    if model_type == "temp":
        model_attr["running state"]["pid"] = 0
        model_attr["running state"]["port"] = model_port
        model_attr["running state"]["host"] = model_host
    elif model_type != "commercial":
        model_attr["running state"]["pid"] = pid
        model_attr["running state"]["port"] = 9009
        model_attr["running state"]["host"] = model_host
    
    # 存入
    model_index = utils.load_json_file(MODEL_INDEX_PATH)
    model_index[model_name] = model_attr
    utils.dump_json_file(model_index,MODEL_INDEX_PATH)
    print(f"[bold blue]已完成 {model_name} 的注册")
    
    #进行测试：
    try:
        inference_model.inference_model(model_name) #内部已经停止
    except Exception as e:
        return False,e
    print(f"[bold blue]已完成 {model_name} 的推理")
    return True,"success"

def insert_model_wrapper(model_name,model_path,model_type,support_vision,model_base,chat_template,port,host):
    log_path = MODEL_ATTR_FOLD / "log" / f"{model_name}.log"  #放在这里是能方便的保证前端也能找到这个文件
    model_port = None
    model_host = "localhost"
    timestamp = utils.generate_timestamp()
    model_path = str(model_path) if model_path else str(MODEL_FOLD / model_name)
    if model_type == "temp":
        model_port = int(port)
        if not model_port:
            raise ValueError(f"forget to write port id!, host name: {host}")
        model_host = host
        model_path = ""
    if chat_template=="default":
        chat_template = "/scratch2/limuyao/workspace/VLA_benchmark/data/model/template/template_llava.jinja"
    flag,message = insert_model(model_name,model_path,model_type,support_vision,model_port,timestamp,log_path,chat_template,model_base,model_host=model_host)
    return flag,message
    

if __name__ == "__main__":
    args = parse_args() #得到参数
    insert_model_wrapper(args.model_name,args.model_path,args.model_type,args.support_vision,args.model_base,args.chat_template,args.port,args.host)
    