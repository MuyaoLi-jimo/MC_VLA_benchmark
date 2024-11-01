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


OPENAI_MODEL = {
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-turbo",
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
    "gpt-4-turbo-browsing",
    "gpt-4-turbo-2024-04-09",
    "gpt2-chatbot",
    "im-also-a-good-gpt2-chatbot",
    "im-a-good-gpt2-chatbot",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4o",
    "gpt-4o-mini",
    "chatgpt-4o-latest-20240903",
    "chatgpt-4o-latest",
    "o1-preview",
    "o1-mini",
}

MODEL_FOLD = Path(__file__).parent.parent.parent / "data" / "model"
MODEL_INDEX_PATH = MODEL_FOLD / "model.json"
MY_MU = 1000
MY_SIGMA = 333.33
MY_BETA = 166.66
MY_TAU = 3.333
GPT_DRAW_P = 0.25
HUMAN_DRAW_P = 0.4
GPT_ENV = trueskill.TrueSkill(mu=MY_MU,sigma=MY_SIGMA,beta=MY_BETA,tau=MY_TAU,draw_probability=GPT_DRAW_P) #TrueSkill的更新参数
HUMAN_ENV = trueskill.TrueSkill(mu=MY_MU,sigma=MY_SIGMA,beta=MY_BETA,tau=MY_TAU,draw_probability=HUMAN_DRAW_P) #TrueSkill的更新参数

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

def run_vllm_server(devices:list,device_num:int,model_path, log_path,port, max_model_len, gpu_memory_utilization,
                    chat_template:str=""):
    
    if devices==[]:
        devices = get_avaliable_gpus(device_num)
    devices_str = ','.join(devices)
    device_num = len(devices)
    
    utils.dump_txt_file("",log_path)
    
    # 构建命令
    
    command = f"CUDA_VISIBLE_DEVICES={devices_str} nohup vllm serve {model_path} --port {port} --max-model-len {max_model_len} --gpu-memory-utilization {gpu_memory_utilization} --trust-remote-code "
    if chat_template!="":
        command += f"--chat-template {chat_template}"
    #if device_num>1:
    #   command += f"--tensor-parallel-size {device_num}"
    command += f" > {log_path} 2>&1 &"
    print(command)
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

def extract_pid(text):
    # Regex pattern to match the line with the PID
    pattern = r"Started server process \[(\d+)\]"
    match = re.search(pattern, text)
    if match:
        return int(match.group(1))  
    return None 

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
        #print(data)
        if data.get("max_new_tokens",None):
            chat_completion = client.chat.completions.create(
                messages=data["messages"],
                model=model,
                max_tokens=data["max_new_tokens"]
                )
        else:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
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
        print(response)
        return response,input_tokens_num,output_tokens_num
    except Exception as e:
        print(f"there is an error：{e}")
        return None

def _create_responce_wrapper(data:dict):
    #id = data["id"]
    #utils.dump_txt_file(data,f"{id}.txt")
    #exit()
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
    """ 
    应该全权负责和model.json的接触
    """
    _MODEL_INDEX_PATH = MODEL_INDEX_PATH

    def __init__(self,model_name,batch_size:int=60):
        
        self._model_index = utils.load_json_file(self._MODEL_INDEX_PATH)
        model_attr = self._model_index[model_name]
        assert model_name in get_avaliable_model_set(self._model_index) or model_name in get_runable_model_set(self._model_index)#avaliable的意思是可以进行评价，runable的意思是可以online测评
        
        self.model_name = model_name
        self.model_id = model_attr["id"]
        self.avaliable = model_attr["avaliable"]
        self.runable = model_attr["runable"]
        self.model_type = model_attr["type"]
        self.model_base = model_attr.get("base","")
        self.support_vision = model_attr["support vision"]
        self.support_text = model_attr.get("support text",True)
        self.chat_template = model_attr.get("template","")
        
        self._model_path = model_attr["path"]
        self.max_token = model_attr.get("max_new_tokens",None)
        self.open_ended_done = set(model_attr.get("OE done",[]))
        self.elo_rating = {
            "model elo rating":model_attr["model elo rating"],
            "human elo rating":model_attr["human elo rating"]
        }
        self.MCQ_score = model_attr["MCQ score"]        
        
        self.running = model_attr["running state"]["running"]    
        self.host = None
        self.port = None
        self.pid =  None
        self.openai_api_base = None
        self.openai_api_key = None
        self.batch_size = batch_size
        
        if self.running and self.model_name not in OPENAI_MODEL:
            self.host = model_attr["running state"]["host"]  
            self.port = model_attr["running state"]["port"]  
            self.pid =  model_attr["running state"]["pid"]  
            self.openai_api_base = f"http://{self.host}:{self.port}/v1"
            self.openai_api_key = "EMPTY"
        
        self.client = None
        self.model = None
        
    def launch(self,device_num = 2, devices=[], host = "localhost",port=9008, max_model_len = 4096, gpu_memory_utilization=0.95):
        """
        TODO:优化。过早的优化是万恶志愿
        如果是线上部署，那转入第一种
        如果不是，那查看是否有pid存入，
            如果有，那说明有模型可以用，直接保存model就好
            如果没有，按照预设的port来创建
        """
        # 如果是线上部署
        if self.model_type == "commercial":
            if self.model_base == "gpt":
                self.openai_api_key = os.getenv("OPENAI_API_KEY")
                if self.openai_api_key == None:
                    print(f"[red]there is not an api key in env")
                    raise EnvironmentError
                self.model = self.model_name
                self.client = OpenAI(
                    api_key=self.openai_api_key,
                )
                print("[yellow]insert the openai-key into the model")
                return
        # 如果是提供的端口
        elif self.model_type=="temp": #如果是临时的
            if not self.runable:
                raise AssertionError("模型已被暂停，无法启动")
            print("[yellow]外部提供的端口，正在运行中，不需要启动")

        # 如果是本地部署
        ## 如果之前已经部署完了，那按照之前的参数走
        else:
            if not self.running:
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
        if self.model_type=="temp":
            print("[yellow]外部提供的端口，不停止")
            return
        if self.model_name in OPENAI_MODEL:
            return
        if not self.running:
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
        if not self.runable:
            raise AssertionError("无法启动")
        for i,data in enumerate(datas):
            if self.model_name not in OPENAI_MODEL:
                data["openai_api_base"] = self.openai_api_base
            else:
                data["model"] = str(self.model)
            if self.max_token:
                data["max_new_tokens"]=self.max_token
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
    
    def get_dataset_responses(self,dataset_name:str):
        dataset_log_path = MODEL_FOLD / "outcome" / self.model_name / f"{dataset_name}.jsonl"
        dataset_lines = utils.load_jsonl(dataset_log_path)
        dataset_dict = {}
        for dataset_line in dataset_lines:
            dataset_dict[dataset_line["id"]] = dataset_line
        return dataset_dict
    
    def get_dataset_elo_rating(self,dataset_name,if_human):
        if if_human:
            key_elo_rating = "human elo rating"
        else:
            key_elo_rating = "model elo rating"
        return self.elo_rating[key_elo_rating].get(dataset_name,{})
        
    def dump_dataset_elo_rating(self,model_elo_rating,dataset_name,if_human):
        if if_human:
            key_elo_rating = "human elo rating"
        else:
            key_elo_rating = "model elo rating"
        self.elo_rating[key_elo_rating][dataset_name] = model_elo_rating
        
    def update_record_running(self,is_running):
        if self.model_base == "gpt": #增加一点鲁棒性
            return
        if is_running:
            self.running=True
            
        else:
            self.running=False
            self.pid = 0
            self.client = None
            self.model = None
            
        self._model_index = utils.load_json_file(self._MODEL_INDEX_PATH)
        self._model_index[self.model_name]["running state"]["running"]=self.running
        self._model_index[self.model_name]["running state"]["pid"]=self.pid
        self._model_index[self.model_name]["running state"]["port"]=self.port
        utils.dump_json_file(self._model_index,self._MODEL_INDEX_PATH)

    def upload_open_end_done(self):
        self._model_index = utils.load_json_file(self._MODEL_INDEX_PATH)
        self._model_index[self.model_name]["OE done"] = list(self.open_ended_done)
        print( self._model_index[self.model_name]["OE done"])
        utils.dump_json_file(self._model_index,self._MODEL_INDEX_PATH)

    def upload_total_elo_rating(self,rating:trueskill.Rating,if_human=False,if_store=True):
        """注意，这里把所有elo指数（包括某task）都更新了，只不过只有trueskill引导的rating是显示更新"""
        if not if_human:
            key_elo_rating = "model elo rating"
        else:
            key_elo_rating = "human elo rating"
        self.elo_rating[key_elo_rating]["total"]["mu"] = int(rating.mu)
        self.elo_rating[key_elo_rating]["total"]["sigma"] = rating.sigma #不要int
        if if_store:
            self.upload_elo_rating(key_elo_rating)
        
    def upload_winrate(self,winrate:float,if_human=False,if_store=True):
        """注意，这里把所有elo指数（包括某task）都更新了，只不过只有trueskill引导的rating是显示更新"""
        if not if_human:
            key_elo_rating = "model elo rating"
        else:
            key_elo_rating = "human elo rating"
        self.elo_rating[key_elo_rating]["total"]["win"] = winrate
        if if_store:
            self.upload_elo_rating(key_elo_rating)
        
    def upload_elo_rating(self,key_elo_rating):
        self._model_index = utils.load_json_file(self._MODEL_INDEX_PATH)
        self._model_index[self.model_name][key_elo_rating] = self.elo_rating[key_elo_rating]
        utils.dump_json_file(self._model_index,self._MODEL_INDEX_PATH)
 
    def _run_vllm_server_in_background(self, devices, max_model_len, gpu_memory_utilization, device_num=1)->int:
        # 日志文件
        
        log_file = MODEL_FOLD / "log" / f"{self.model_name}.log"
        
        pid,devices_str = run_vllm_server(devices,device_num,self._model_path,log_file,port=self.port,max_model_len=max_model_len,gpu_memory_utilization=gpu_memory_utilization,
                                          chat_template=self.chat_template)
            
        print(f"Started {self.model_name} on GPU {devices_str} with PID {pid}. Logs are being written to {log_file}")
        return pid
        
def get_avaliable_model_set(model_index:dict={},type="total",dataset=""):
    """得到当前所有可用的数据 """
    if model_index=={}:
        model_index_path = Path(__file__).parent.parent.parent / "data" / "model" / "model.json"
        model_index = utils.load_json_file(model_index_path)
    avaliable_models = set()
    for model_name in model_index.keys():
        if model_index[model_name]["avaliable"] and (type == "total" or model_index[model_name]["type"] == type) and (not dataset or dataset in set(model_index[model_name]["OE done"])):
            avaliable_models.add(model_name)
    return avaliable_models

def get_avaliable_model_dict(model_index:dict={},type="total",dataset=""):
    avaliable_models = {}
    avaliable_models_set = get_avaliable_model_set(model_index=model_index,type=type,dataset=dataset)
    for avaliable_model in avaliable_models_set:
        avaliable_models[avaliable_model] = Model(model_name=avaliable_model)
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
 
def get_model_ratings(model_index:dict={},type="total",if_human=False):
    env = None
    if not if_human:
        key_elo_rating = "model elo rating"
        env = GPT_ENV
    else:
        key_elo_rating = "human elo rating"
        env = HUMAN_ENV
    if model_index=={}:
        model_index_path = Path(__file__).parent.parent.parent / "data" / "model" / "model.json"
        model_index = utils.load_json_file(model_index_path)
    avaliable_models = get_avaliable_model_set(model_index)
    model_ratings = {}
    for avaliable_model in avaliable_models:
        model_ratings[avaliable_model] = env.create_rating(model_index[avaliable_model][key_elo_rating]["total"]["mu"],model_index[avaliable_model][key_elo_rating]["total"]["sigma"])
    return model_ratings
        
if __name__ == "__main__":
    #model_ratings = get_model_ratings()
    #print(type(model_ratings["gpt-4o-mini"].sigma))
    print(get_avaliable_model_set(dataset="visual-basic"))
    #model = Model(model_name="llama3-llava-next-8b-hf")
    #model.launch(devices=["3"])
    #time.sleep(10)
    #model.stop()