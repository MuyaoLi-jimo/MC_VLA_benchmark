from openai import OpenAI
from rich import print
import subprocess
import multiprocessing as mp
import os
import signal
import time
import re
import copy
from pathlib import Path
from utils import utils
import pickle

OPENAI_MODEL = [
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06", #supports Structured Outputs
    "chatgpt-4o-latest",
    "gpt-4o-mini",
    "gpt-4-1106-preview",
    "gpt-3.5-turbo-0125",
]


def get_gpu_usage():
    """check usage off gpu"""
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'], 
                            stdout=subprocess.PIPE, 
                            encoding='utf-8')
    gpu_info = result.stdout.strip().split('\n')
    
    gpu_usages = []
    for idx, info in enumerate(gpu_info):
        used, total = map(int, info.split(','))
        gpu_usages.append((idx, used, total))
    
    return gpu_usages

def get_avaliable_gpus(cuda_num:int):
    avaliable_gpus = []
    gpu_usages = get_gpu_usage()
    for gpu_usage in gpu_usages:
        idx, used, total = gpu_usage
        if used/total < 0.1:
            avaliable_gpus.append(str(idx))
            if len(avaliable_gpus)>=cuda_num:
                break
    if len(avaliable_gpus)<cuda_num:
        print("there aren't enough avaliable GPUs to use")
        print(f"[red]{gpu_usages}")
        raise Exception
    return avaliable_gpus

def _create_responce(data:dict):
    model= None
    client = None
    if data.get("model","") in OPENAI_MODEL:
         client = OpenAI(
             #api_key = data["openai_api_key"],
         )
         model = data["model"]
    else:
        client = OpenAI(
            api_key=data["openai_api_key"],
            base_url=data["openai_api_base"],
        )
        models = client.models.list()
        model = models.data[0].id

    try:                 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
        chat_completion = client.chat.completions.create(
            messages=data["messages"],
            model=model)
        #print(chat_completion)dict(chat_completion.choices[0].message)
        response = {
            "role":str(chat_completion.choices[0].message.role),
            "content":str(chat_completion.choices[0].message.content),
        }
        input_tokens_num = chat_completion.usage.prompt_tokens
        output_tokens_num = chat_completion.usage.completion_tokens
        return response,input_tokens_num,output_tokens_num
    except Exception as e:
        print(e)
        return None

def _create_responce_wrapper(data:dict):
    response,input_tokens_num,output_tokens_num = _create_responce(data)
    return {"id":data["id"],"message":response,"input_tokens":input_tokens_num,"output_tokens":output_tokens_num}
    
def _create_chunk_responces(datas:list):
    num_processes = len(datas)
    results = []
    if num_processes > 1:
        pool = mp.Pool(processes=num_processes) 
        #print(datas)  
        results = pool.map(_create_responce_wrapper, datas)  
        pool.close() # close the pool
        pool.join()
    else:
        results = [_create_responce_wrapper(datas[0])]
        
    return results

class Model:
    LEGAL_MODEL = [
        "llama3-llava-next-8b-hf",
        "gpt-4o-mini",
        "gpt-4o",
    ]
    MODEL_FOLD = Path(__file__).parent.parent.parent / "data" / "model"
    MODEL_INDEX_PATH = MODEL_FOLD / "model.json"
    def __init__(self,model_name,model_fold = "/nfs-shared/models",batch_size:int=60):
        assert model_name in self.LEGAL_MODEL
        self.model_index = utils.load_json_file(self.MODEL_INDEX_PATH)
        self.model_name = model_name
        self.model_attr = self.model_index[model_name]
        self.model_path = self.model_attr["path"]
        
        self.host = None
        self.port = None
        self.pid =  None
        self.openai_api_base = None
        self.openai_api_key = None
        if self.model_attr["running"] and self.model_name not in OPENAI_MODEL:
            self.host = self.model_attr["host"]
            self.port = self.model_attr["port"]
            self.pid =  self.model_attr["pid"]
            self.openai_api_base = f"http://{self.host}:{self.port}/v1"
            self.openai_api_key = "EMPTY"
        
        self.client = None
        self.model = None
        
        self.batch_size = batch_size
        
    def launch(self,device_num = 2, devices=[], host = "localhost",port=9001, max_model_len = 4096, gpu_memory_utilization=0.95):
        """
        如果是线上部署，那转入第一种
        如果不是，那查看是否有pid存入，
            如果有，那说明有模型可以用，直接保存model就好
            如果没有，按照预设的port来创建
        """
        # 如果是线上部署
        if self.model_name in OPENAI_MODEL:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            if self.openai_api_key == None:
                print(f"[red]there is not an api key in env")
                raise EnvironmentError
            else:
                self.model = self.model_name
                self.client = OpenAI(
                    api_key=self.openai_api_key,
                )
                print("[yellow]insert the openai-key into the model")
            return
        
        # 如果是本地部署
        ## 如果之前已经部署完了，那按照之前的参数走
        if self.model_attr["running"]:
            if type(self.model) == type(None):   
                self.client = OpenAI(
                    api_key=self.openai_api_key,
                    base_url=self.openai_api_base,
                )
                models = self.client.models.list()
                self.model = models.data[0].id
        else:
            self.port = port
            self.host = host
            if devices==[]:
                devices = get_avaliable_gpus(device_num)
            self.pid = self._run_vllm_server_in_background(
                gpu_devices=devices,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization
            )
            self.update_record_running(True)
            self.openai_api_base = f"http://{self.host}:{self.port}/v1"
            self.openai_api_key = "EMPTY"
    
            self.client = OpenAI(
                api_key=self.openai_api_key,
                base_url=self.openai_api_base,
            )
            models = self.client.models.list()
            self.model = models.data[0].id
            
        print(f"[green]Model Launched, pid: {self.pid}, host: {self.host}, port: {self.port}")
        
        return
    
    def stop(self):
        if self.model_name in OPENAI_MODEL:
            return
        if not self.model_attr["running"]:
            print(f"[green] Model {self.model_name} not in process")
        
        os.kill(self.pid, signal.SIGINT)
        self.update_record_running(is_running=False)
        
        print(f"[red]Stop {self.pid} service {self.model_name}")
    
    def inference(self,datas,batch_size:int=60):
        """_summary_
        Args:
            datas (list): id,image,content or id content
            type (str): the type of inference
        """
        for i,data in enumerate(datas):
            if self.model_name not in OPENAI_MODEL:
                data["openai_api_base"] = self.openai_api_base
            else:
                data["model"] = str(self.model)
            data["openai_api_key"] = str(self.openai_api_key)
            
            datas[i] = data
        results = []
        data_len = len(datas)
        self.batch_size = batch_size
        chunk_size = data_len//self.batch_size if data_len % self.batch_size==0 else data_len//self.batch_size + 1
        for i in range(chunk_size):
            if (i+1) * self.batch_size <= data_len:
                chunk_datas = datas[i * self.batch_size: (i + 1) * self.batch_size]
            else:
                chunk_datas = datas[i*self.batch_size:data_len]
            
            chunk_results = _create_chunk_responces(chunk_datas)
            results.extend(chunk_results)
        return results
    
    def update_record_running(self,is_running):
        if self.model_name in OPENAI_MODEL: #增加一点鲁棒性
            return
        if is_running:
            self.model_attr["running"]=True
            self.model_attr["pid"]=self.pid
            self.model_attr["host"]=self.host
            self.model_attr["port"]=self.port
        else:
            self.pid = None
            self.client = None
            self.model = None
            self.model_attr["running"]=False
            self.model_attr["pid"]=0
            
        self.model_index[self.model_name] = self.model_attr
        utils.dump_json_file(self.model_index,self.MODEL_INDEX_PATH)
    
    def _run_vllm_server_in_background(self, gpu_devices, max_model_len, gpu_memory_utilization):
        # 日志文件
        
        log_file = self.MODEL_FOLD / "log" / f"{self.model_name}.log"
        
        gpu_devices_str = ','.join(gpu_devices)
        print(gpu_devices_str)
        
        utils.dump_txt_file("",log_file)
        
        # 构建命令
        command = f"CUDA_VISIBLE_DEVICES={gpu_devices_str} nohup vllm serve {self.model_path} --port {self.port} --max-model-len {max_model_len} --gpu-memory-utilization {gpu_memory_utilization} > {log_file} 2>&1 &"
        
        # 使用 shell=True 来运行 nohup 命令
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 获取进程的 PID
        while True:
            log_text = utils.load_txt_file(log_file)
            pid = extract_pid(log_text)
            if type(pid) != type(None):
                break
            print("[yellow]not found server")
            time.sleep(10)
            
        print(f"Started {self.model_name} on GPU {gpu_devices} with PID {pid}. Logs are being written to {log_file}")
        return pid
        
def extract_pid(text):
    # Regex pattern to match the line with the PID
    pattern = r"Started server process \[(\d+)\]"
    match = re.search(pattern, text)
    if match:
        return int(match.group(1))  
    return None 
        
if __name__ == "__main__":
    model = Model(model_name="llama3-llava-next-8b-hf")
    model.launch(devices=["3"])
    time.sleep(10)
    model.stop()