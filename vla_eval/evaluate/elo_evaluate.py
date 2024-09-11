"""
elo evaluating: 
    step0: 输入模型A，dataset
    step1: 按照要求/随机从所有model中 拿出一个model B
    step2: A和B 同时作答dataset中的问题，并把结果记录到文件中
    step3: judge 对A和B的得分进行评价
    step4: 计算elo分数  -- 当前改用true skill的参数
        有以下几个elo指标需要计算： 总elo指标、一个dataset的elo指标、 一个task的elo指标，
        还有两种模式：detailed: 按照每一个问题来计算，broad：一个dataset计算一个
    附加任务：把这次的比赛记录起来，要求记录：modelA，modelB，judge，token，timestamp,dataset，
        
@author limuyao
"""
import numpy as np
import multiprocessing as mp
from rich import print
from pathlib import Path
from tqdm import tqdm
import trueskill 
from vla_eval.evaluate import elo_validate,inference
from vla_eval.dataset import dataset_wrapper
from vla_eval.dataset.base import BaseDataset
from vla_eval.model import model,rank_model
from utils import utils

DATA_FOLD = Path(__file__).parent.parent.parent / "data"
MODEL_PATH = DATA_FOLD / "model" / "model.json"
HISTORY_PATH = DATA_FOLD / "history.jsonl"
HUMAN_HISTORY_PATH = DATA_FOLD / "human_history.jsonl"
ROUND_NUM = 5   #一场比赛寻找好对手的迭代次数
ROUND_QUA = 0.3 #定义好比赛
STOP_A = 5
K = 32  #elo公式的更新参数

def offline_elo_evaluate(round_time=100,model_A_name:str="",judge_model_name="gpt-4o"):
    timestamp = utils.generate_timestamp()
    model_ratings = model.get_model_ratings()
    history_jp = utils.JsonlProcessor(HISTORY_PATH)
    judge_model = model.Model(judge_model_name)
    for i in tqdm(range(round_time)):
        if i%5==0 and i>20:
            stop_flag=True
            for model_name in model_ratings.keys():
                if model_ratings[model_name].sigma > STOP_A:
                    print(f"{model_name}没有收敛，目前标准差{model_ratings[model_name].sigma}")
                    stop_flag = False
                    break
            if stop_flag:
                break
        # sample出一个组合
        model_A_name,model_B_name,dataset_name = sample_A_B_D(model_ratings,model_A_name=model_A_name)
        # 随机从A和B中抽一个
        dataset = dataset_wrapper.make(dataset_name)
        model_A = model.Model(model_A_name)
        model_B = model.Model(model_B_name)
        outcome = None
        if np.random.choice([True,False]):
            outcome = elo_validate.offline_validate(dataset,model_A,model_B,judge_model)
            model_ratings,model_A,model_B,dataset = cal_elo(outcome,model_ratings,model_A,model_B,dataset)  #其实没必要这么写，但是闲的
        else:
            outcome = elo_validate.offline_validate(dataset,model_B,model_A,judge_model)
            model_ratings,model_B,model_A,dataset = cal_elo(outcome,model_ratings,model_B,model_A,dataset)
        outcome.update({"timestamp":timestamp})
        history_jp.dump_line(outcome)
        
    print("done")

def history_elo_evaluate(choice:str = "total",if_print_elo=False,if_human=False):
    """由历史记录重新计算所有模型的elo参数(只计算总榜), 顺带计算平局概率，胜率"""
    if if_human:
        history_jp = utils.JsonlProcessor(HUMAN_HISTORY_PATH)
    else:
        history_jp = utils.JsonlProcessor(HISTORY_PATH)
    model_set = model.get_avaliable_model_set()
    model_rating = {}
    model_win_rate = {}
    for model_name in model_set:
        model_rating[model_name] = init_trueskill_rating()
        model_win_rate[model_name] = {
            "total":0,
            "win_tie":0,
        }
    draw_num = 0
    while True:
        line = history_jp.load_line()
        if type(line) == type(None):
            break
        model_A_name = line["model_A"]
        model_B_name = line["model_B"]
        score = line["score"]
        dataset_name = line["dataset"]
        if score==2:
            draw_num+=1
        model_win_rate = cal_win_rate(score,model_A_name,model_B_name,model_win_rate)
        model_rating[model_A_name],model_rating[model_B_name] = cal_trueskill_rating(score,model_rating[model_A_name],model_rating[model_B_name])
    total_num = history_jp.len()
    if total_num!=0:
        print(f"[cyan]平局数: {draw_num/total_num}")
    print(f"{model_rating}")
    
    
    def upload_elo(model_rating:dict,if_human=False):
        if not if_human:
            key_elo_rating = "elo rating"
        else:
            key_elo_rating = "human rating"
        model_file = utils.load_json_file(MODEL_PATH)
        for model_name in model_set:
            model_file[model_name][key_elo_rating]["total"]["mu"] = int(model_rating[model_name].mu)
            model_file[model_name][key_elo_rating]["total"]["sigma"] = model_rating[model_name].sigma
        utils.dump_json_file(model_file,MODEL_PATH)
        
    def upload_win_rate(model_win_rate:dict,if_human=False):
        if not if_human:
            key_elo_rating = "elo rating"
        else:
            key_elo_rating = "human rating"
        model_file = utils.load_json_file(MODEL_PATH)
        for model_name in model_set:
            if model_win_rate[model_name]["total"]==0:
                model_file[model_name][key_elo_rating]["total"]["win"]=0
            else:
                model_file[model_name][key_elo_rating]["total"]["win"] = model_win_rate[model_name]["win_tie"]/model_win_rate[model_name]["total"]
        utils.dump_json_file(model_file,MODEL_PATH)
        
    upload_elo(model_rating,if_human)
    upload_win_rate(model_win_rate,if_human)
    rank_pd = rank_model.elo_rank(choice=choice,if_print_elo=if_print_elo,if_human=if_human)
    return rank_pd

def online_elo_evaluate(dataset_name:str, model_A_name:str,model_B_name:str="",judge_model_name = "gpt-4o",motion="detailed"):
    raise AssertionError("未更新，无法使用")
    timestamp = utils.generate_timestamp()
    dataset = dataset_wrapper.make(dataset_name)
    if model_B_name == "":
        model_B_name = sample_B_model(model_A_name,dataset)
    # 启动！
    model_A = model.Model(model_A_name)
    model_B = model.Model(model_B_name)
    judge = model.Model(judge_model_name)
    print(f"[red]start evaluate, model A:{model_A_name}, model B: {model_B_name}, dataset: {dataset_name}")
    model_A.launch(device_num=1,port=9010)
    model_B.launch(device_num=1,port=9011)
    judge.launch(device_num=1,port=9012)
    # 运行两个模型
    p_A = run_inference(dataset,model_A,timestamp)
    p_B = run_inference(dataset,model_B,timestamp)
    # 评估表现
    validate_outcome = elo_validate.online_validate(dataset,model_A,model_B,timestamp,judge)
    # 计算得分
    cal_elos(validate_outcome["score"],model_A,model_B,dataset=dataset,motion=motion)
    
    # 记录
    history = {
        "timestamp":timestamp,
        "dataset":dataset.dataset_name,
        "model_A":model_A.model_name,
        "model_B":model_B.model_name,
        "judge":judge.model_name,
        "score":validate_outcome["score"],
        "token":validate_outcome["token"],
    }
    history_jp = utils.JsonlProcessor(HISTORY_PATH)
    history_jp.dump_line(history)
    print("完成记录")
    # 打扫打扫
    history_jp.close()
    p_A.join()
    p_B.join()
    model_A.stop()
    model_B.stop()
    judge.stop()
    
def run_inference(database:BaseDataset,inference_model:model.Model,timestamp):
    p = mp.Process(target=inference.inference,args=(database,inference_model,timestamp))
    p.start()
    return p
        
def sample_A_B_D(model_ratings:dict,model_A_name:str = "",model_B_name = "",dataset_name="", model_file:dict={}):
    "从model_rating中选出一组高质量的对局" 
    # 首先按照标准差来选择A：
    if model_file=={}:
        model_file = utils.load_json_file(MODEL_PATH)
    avaliable_models = list(model_ratings.keys())
    model_sigma = [model_ratings[avaliable_model].sigma for avaliable_model in avaliable_models ]
    model_prob = np.exp(model_sigma) / np.sum(np.exp(model_sigma))
    if not model_A_name:
        model_A_name = np.random.choice(avaliable_models, size=1, p=model_prob)[0]
    if not dataset_name:
        dataset_name = np.random.choice(model_file[model_A_name]["done"])
    if not model_B_name:
        for _ in range(ROUND_NUM): #超参数，最大尝试次数
            model_B_name = sample_B_model(model_A_name,dataset_name,model_file)
            # 看看是不是好的比赛
            quality = trueskill.quality_1vs1(model_ratings[model_A_name],model_ratings[model_B_name],env=model.MY_ENV)
            if quality > ROUND_QUA:
                break
    return model_A_name,model_B_name,dataset_name
        
def sample_B_model(model_A_name:str,dataset_name:str,model_file:dict={}):
    if model_file=={}:
        model_file = utils.load_json_file(MODEL_PATH)
    avaliable_model = model.get_avaliable_model_set(model_file)
    model_set = set()
    for model_name in avaliable_model:
        if model_name == model_A_name:
            continue
        if dataset_name not in set(model_file[model_name]["done"]):
            continue        
        model_set.add(model_name)
    model_B_name = np.random.choice(list(model_set))
    return model_B_name

def estimated_win_loss_rate(A_rating,B_rating):
    """由A和B的rating来计算胜率 """
    e_A = 1/(1+10**((B_rating-A_rating)/400))
    return e_A

def update_elo_rating(A_rating,B_rating,outcome):
    if outcome==4:  #在这里差当做平局处理
        outcome = 2
    e_A = estimated_win_loss_rate(A_rating,B_rating)
    s_A = (outcome - 1) / 2
    update_A = K*(s_A-e_A)
    new_A_rating,new_B_rating = A_rating + update_A, B_rating - update_A
    return int(new_A_rating), int(new_B_rating)

def init_trueskill_rating():
    return model.MY_ENV.create_rating()

def init_elo_rating(model_elo_rating:dict,dataset:BaseDataset,my_task:str="total"):
    if "total" not in model_elo_rating:
        model_elo_rating["total"] = 1000
    if my_task == "total":
        tasks = dataset.get_tasks()
        for task in tasks:
            if task not in model_elo_rating:
                model_elo_rating[task] = 1000
    else:
        if my_task not in model_elo_rating:
            model_elo_rating[my_task] = 1000
    return model_elo_rating

def cal_trueskill_rating(score,rate_A:trueskill.Rating,rate_B:trueskill.Rating):
    if score==3: #赢了
        rate_A,rate_B = trueskill.rate_1vs1( rate_A,rate_B,env=model.MY_ENV)
    elif score==2 or score==4: #平
        rate_A,rate_B = trueskill.rate_1vs1( rate_A,rate_B,drawn=True,env=model.MY_ENV)
    elif score==1: #输了
        rate_B,rate_A = trueskill.rate_1vs1( rate_B,rate_A,env=model.MY_ENV)
    else:
        print(f"score错误：{score}")
    return rate_A,rate_B

def cal_elo(outcome:dict,model_rating:dict,model_A:model.Model,model_B:model.Model,dataset:BaseDataset,if_human = False):
    if if_human:
        key_elo_rating = "human rating"
    else:
        key_elo_rating = "elo rating"
        
    model_A_elo_rating = model_A.model_attr[key_elo_rating].get(dataset.dataset_name,{})
    model_B_elo_rating = model_B.model_attr[key_elo_rating].get(dataset.dataset_name,{})
    question_id = outcome["id"]
    task = dataset.get_task(question_id)
    score = outcome["score"]
    model_A_elo_rating = init_elo_rating(model_A_elo_rating,dataset=dataset,my_task=task)
    model_B_elo_rating = init_elo_rating(model_B_elo_rating,dataset=dataset,my_task=task)

    model_rating[model_A.model_name],model_rating[model_B.model_name] = cal_trueskill_rating(score,model_rating[model_A.model_name],model_rating[model_B.model_name])
    
    model_A_elo_rating["total"],model_B_elo_rating["total"] = update_elo_rating(model_A_elo_rating["total"],model_B_elo_rating["total"],score)
    model_A_elo_rating[task],   model_B_elo_rating[task]    = update_elo_rating(model_A_elo_rating[task],model_B_elo_rating[task],score)
    model_A.model_attr[key_elo_rating][dataset.dataset_name] = model_A_elo_rating
    model_B.model_attr[key_elo_rating][dataset.dataset_name] = model_B_elo_rating
    
    model_A.upload_elo_rating(model_rating[model_A.model_name],if_human)
    model_B.upload_elo_rating(model_rating[model_B.model_name],if_human)
    
    return model_rating,model_A,model_B,dataset
    
def cal_elos(scores:dict,model_A:model.Model,model_B:model.Model,dataset:BaseDataset,motion:str):
    """更新模型整体的elo rating数值
    """
    raise Exception("暂未更新，无法使用")
    task_scores = {}
    total_scores = []
    dataset_name = dataset.dataset_name
    for id in scores.keys():
        total_scores.append(scores[id])
        task = dataset.get_task(id)
        if task in task_scores:
            task_scores[task].append(scores[id])
        else:
            task_scores[task] = [scores[id]]
    
    model_A_elo_rating = model_A.model_attr["elo rating"]
    model_B_elo_rating = model_B.model_attr["elo rating"]
    
    model_A_elo_rating = init_elo_rating(model_A_elo_rating,dataset=dataset)
    model_B_elo_rating = init_elo_rating(model_B_elo_rating,dataset=dataset)
    
    if motion == "broad":
        total_score = np.mean(total_scores) # 计算平均成绩
        # 计算总成绩
        model_A_elo_rating["total"],model_B_elo_rating["total"] = update_elo_rating(model_A_elo_rating["total"],model_B_elo_rating["total"],total_score)
        # 计算该dataset的总成绩
        model_A_elo_rating[dataset_name]["total"],model_B_elo_rating[dataset_name]["total"] = update_elo_rating(model_A_elo_rating[dataset_name]["total"],model_B_elo_rating[dataset_name]["total"],total_score)
        # 计算该dataset每个task的总成绩
        for task,value in task_scores.items():
            task_score = np.mean(value)
            model_A_elo_rating[dataset_name][task],model_B_elo_rating[dataset_name][task] = update_elo_rating(model_A_elo_rating[dataset_name][task],model_B_elo_rating[dataset_name][task],task_score)
    elif motion == "detailed":
        for score in total_scores:
            model_A_elo_rating["total"],model_B_elo_rating["total"] = update_elo_rating(model_A_elo_rating["total"],model_B_elo_rating["total"],score)
            model_A_elo_rating[dataset_name]["total"],model_B_elo_rating[dataset_name]["total"] = update_elo_rating(model_A_elo_rating[dataset_name]["total"],model_B_elo_rating[dataset_name]["total"],score)
        for task,value in task_scores.items():
            for score in value:
                model_A_elo_rating[dataset_name][task],model_B_elo_rating[dataset_name][task] = update_elo_rating(model_A_elo_rating[dataset_name][task],model_B_elo_rating[dataset_name][task],score)
    else:
        raise AssertionError
    
    #print(model_A_elo_rating)
    #print(model_B_elo_rating)
    model_A.model_attr["elo rating"] = model_A_elo_rating
    model_B.model_attr["elo rating"] = model_B_elo_rating
    model_A.upload_model_attr()
    model_B.upload_model_attr()

def cal_win_rate(score:int,model_A_name:str,model_B_name:str,model_win_rate:dict):
    model_win_rate[model_A_name]["total"]+=1
    model_win_rate[model_B_name]["total"]+=1
    if score==3:
        model_win_rate[model_A_name]["win_tie"]+=1
    elif score==2:
        model_win_rate[model_A_name]["win_tie"]+=1
        model_win_rate[model_B_name]["win_tie"]+=1
    elif score==1:
        model_win_rate[model_B_name]["win_tie"]+=1
    return model_win_rate


if __name__ == "__main__":
    #offline_elo_evaluate(model_A_name="")
    history_elo_evaluate()
    #model_ratings = model.get_model_ratings()
    #print(sample_A_B_D(model_ratings))
    #dataset = dataset_wrapper.make("knowledge")
    #a = sample_B_model("10003-2024-09-02 09:46:53",dataset)
    #print(a)
    #Rating = model.MY_ENV.create_rating(mu=1309)
    #print(Rating)

    #elo_evaluate(dataset_name="knowledge",model_A_name="llama3-llava-next-8b-hf",model_B_name="gpt-4o-mini")


