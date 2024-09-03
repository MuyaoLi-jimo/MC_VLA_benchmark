from openai import OpenAI
from rich import print
import subprocess
import multiprocessing as mp
import os
import signal
import time
import re
import trueskill 
from pathlib import Path
from utils import utils
import abc

OPENAI_MODEL = [
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06", #supports Structured Outputs
    "chatgpt-4o-latest",
    "gpt-4o-mini",
    "gpt-4-1106-preview",
    "gpt-3.5-turbo-0125",
]

MODEL_FOLD = Path(__file__).parent.parent.parent / "data" / "model"
MODEL_INDEX_PATH = MODEL_FOLD / "model.json"
MY_MU = 1000
MY_SIGMA = 333.33
MY_BETA = 166.66
MY_TAU = 3.333
MY_DRAW_P = 0.1
MY_ENV = trueskill.TrueSkill(mu=MY_MU,sigma=MY_SIGMA,beta=MY_BETA,tau=MY_TAU,draw_probability=MY_DRAW_P) #TrueSkill的更新参数


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
        if used/total < 0.03:
            avaliable_gpus.append(str(idx))
            if len(avaliable_gpus)>=cuda_num:
                break
    if len(avaliable_gpus)<cuda_num:
        print("there aren't enough avaliable GPUs to use")
        print(f"[red]{gpu_usages}")
        raise Exception
    return avaliable_gpus

def run_vllm_server(devices:list,device_num:int,model_path, log_path,port, max_model_len, gpu_memory_utilization):
    
    if devices==[]:
        devices = get_avaliable_gpus(device_num)
    devices_str = ','.join(devices)
    
    utils.dump_txt_file("",log_path)
    
    # 构建命令
    command = f"CUDA_VISIBLE_DEVICES={devices_str} nohup vllm serve {model_path} --port {port} --max-model-len {max_model_len} --gpu-memory-utilization {gpu_memory_utilization} --trust-remote-code > {log_path} 2>&1 &"
    
    # 使用 shell=True 来运行 nohup 命令
    subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
            # 获取进程的 PID,使用这个方法是为了保证vllm加载的模型已经开始运行
    while True:
        log_text = utils.load_txt_file(log_path)
        pid = extract_pid(log_text)
        if type(pid) != type(None):
            break
        print("[yellow]not found server yet")
        time.sleep(10)
    
    return int(pid),devices_str

def stop_vllm_server(pid:int):
    os.kill(pid, signal.SIGINT)

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

def extract_pid(text):
    # Regex pattern to match the line with the PID
    pattern = r"Started server process \[(\d+)\]"
    match = re.search(pattern, text)
    if match:
        return int(match.group(1))  
    return None 

class Model:
    _MODEL_FOLD = MODEL_FOLD
    _MODEL_INDEX_PATH = MODEL_INDEX_PATH
    def __init__(self,model_name,batch_size:int=80):
        
        self.model_index = utils.load_json_file(self._MODEL_INDEX_PATH)
        assert model_name in get_avaliable_model_set(self.model_index) #avaliable的意思是可以进行评价，runable的意思是可以online测评
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
        
    def launch(self,device_num = 2, devices=[], host = "localhost",port=9008, max_model_len = 4096, gpu_memory_utilization=0.95):
        """
        如果是线上部署，那转入第一种
        如果不是，那查看是否有pid存入，
            如果有，那说明有模型可以用，直接保存model就好
            如果没有，按照预设的port来创建
        """
        # 如果是提供的端口
        if self.model_attr["type"]=="temp": #如果是临时的
            if not self.model_attr["runable"]:
                print("模型已被暂停，无法启动")
                return
            self.openai_api_base = f"http://{self.host}:{self.port}/v1"
            self.openai_api_key = "EMPTY"
            print("外部提供的端口，正在运行中，不干涉")
            self.client = OpenAI(
                api_key=self.openai_api_key,
                base_url=self.openai_api_base,
            )
            models = self.client.models.list()
            self.model = models.data[0].id
            return
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
            self.pid = self._run_vllm_server_in_background(
                devices=devices,
                device_num=device_num,
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
        if self.model_attr["type"]=="temp":
            print("外部提供的端口，不干涉")
            return
        if self.model_name in OPENAI_MODEL:
            return
        if not self.model_attr["running"]:
            print(f"[green] Model {self.model_name} not in process")
        pid = self.pid
        stop_vllm_server(pid=pid)
        self.update_record_running(is_running=False)
        
        print(f"[red]Stop {pid} service {self.model_name}")
    
    def inference(self,datas,batch_size:int=80):
        """_summary_
        Args:
            datas (list): id,image,content or id content
            type (str): the type of inference
        """
        if not self.model_attr["runable"]:
            raise AssertionError("无法启动")
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
        self.upload_model_attr()
    
    def upload_elo_rating(self,rating:trueskill.Rating):
        self.model_attr["elo rating"]["total"]["mu"] = int(rating.mu)
        self.model_attr["elo rating"]["total"]["sigma"] = int(rating.sigma)
        self.upload_model_attr()
    
    def upload_model_attr(self):
        """更新attr """
        self.model_index = utils.load_json_file(self._MODEL_INDEX_PATH) #!锁！！！什么时候给加上？
        self.model_index[self.model_name] = self.model_attr
        utils.dump_json_file(self.model_index,self._MODEL_INDEX_PATH)
      
    def get_dataset_responses(self,dataset_name:str):
        dataset_log_path = MODEL_FOLD / "outcome" / self.model_name / f"{dataset_name}.jsonl"
        dataset_lines = utils.load_jsonl(dataset_log_path)
        dataset_dict = {}
        for dataset_line in dataset_lines:
            dataset_dict[dataset_line["id"]] = dataset_line
        return dataset_dict
     
    def _run_vllm_server_in_background(self, devices, max_model_len, gpu_memory_utilization, device_num=1)->int:
        # 日志文件
        
        log_file = self._MODEL_FOLD / "log" / f"{self.model_name}.log"
        
        pid,devices_str = run_vllm_server(devices,device_num,self.model_path,log_file,port=self.port,max_model_len=max_model_len,gpu_memory_utilization=gpu_memory_utilization)
            
        print(f"Started {self.model_name} on GPU {devices_str} with PID {pid}. Logs are being written to {log_file}")
        return pid
        
def get_avaliable_model_set(model_index:dict={},type="total"):
    """得到当前所有可用的数据 """
    if model_index=={}:
        model_index_path = Path(__file__).parent.parent.parent / "data" / "model" / "model.json"
        model_index = utils.load_json_file(model_index_path)
    avaliable_models = set()
    for model_name in model_index.keys():
        if model_index[model_name]["avaliable"] and (type == "total" or model_index[model_name]["type"] == type):
            avaliable_models.add(model_name)
        
    return avaliable_models
 
def get_runable_model_set(model_index:dict={},type="total"):
    if model_index=={}:
        model_index_path = Path(__file__).parent.parent.parent / "data" / "model" / "model.json"
        model_index = utils.load_json_file(model_index_path)
    runable_models = set()
    for model_name in model_index.keys():
        if model_index[model_name]["runable"] and (type == "total" or model_index[model_name]["type"] == type):
            runable_models.add(model_name)
    return runable_models
 
def get_model_ratings(model_index:dict={},type="total"):
    if model_index=={}:
        model_index_path = Path(__file__).parent.parent.parent / "data" / "model" / "model.json"
        model_index = utils.load_json_file(model_index_path)
    avaliable_models = get_avaliable_model_set(model_index)
    model_ratings = {}
    for avaliable_model in avaliable_models:
        model_ratings[avaliable_model] = MY_ENV.create_rating(model_index[avaliable_model]["elo rating"]["total"]["mu"],model_index[avaliable_model]["elo rating"]["total"]["sigma"])
    return model_ratings
        
if __name__ == "__main__":
    model_ratings = get_model_ratings()
    print(type(model_ratings["gpt-4o-mini"].sigma))
    
    #model = Model(model_name="llama3-llava-next-8b-hf")
    #model.launch(devices=["3"])
    #time.sleep(10)
    #model.stop()