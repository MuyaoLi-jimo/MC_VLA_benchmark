from utils import utils
from vla_eval.model import model
from rich import print
from copy import deepcopy
from pathlib import Path
DATA_FOLD = Path(__file__).parent.parent.parent / "data"
HUMAN_HISTORY_PATH = DATA_FOLD / "human_history_database"
HISTORY_PATH = DATA_FOLD / "history.jsonl"


def get_LLM_score(Model_score:dict):
    history_jp = utils.JsonlProcessor(HISTORY_PATH)
    while True:
        line = history_jp.load_line()
        if not line:
            break
        model_A_name = line["model_A"]
        model_B_name = line["model_B"]
        dataset_name = line["dataset"]
        score = line["score"]
        if score in {4,0}:
            continue
        if (model_A_name,model_B_name,dataset_name) in Model_score:
            Model_score[(model_A_name,model_B_name,dataset_name)][1] += 1
            Model_score[(model_A_name,model_B_name,dataset_name)][0] += score
        elif (model_B_name,model_A_name,dataset_name) in Model_score:
            Model_score[(model_B_name,model_A_name,dataset_name)][1] += 1
            Model_score[(model_B_name,model_A_name,dataset_name)][0] += 4-score
    return Model_score

def get_Human_score(Model_score:dict):
    history_db = utils.LmdbProcessor(HUMAN_HISTORY_PATH)
    history_info = history_db.get_info()
    for _,value in history_info.items():
        model_A_name = value["model_A"]
        model_B_name = value["model_B"]
        dataset_name = value["dataset"]
        score = value["score"]
        if score in {4,0}:
            continue
        if (model_A_name,model_B_name,dataset_name) in Model_score:
            Model_score[(model_A_name,model_B_name,dataset_name)][1] += 1
            Model_score[(model_A_name,model_B_name,dataset_name)][0] += score
        elif (model_B_name,model_A_name,dataset_name) in Model_score:
            Model_score[(model_B_name,model_A_name,dataset_name)][1] += 1
            Model_score[(model_B_name,model_A_name,dataset_name)][0] += 4-score
    return Model_score


def cal_pearson(LLM_score:dict,Human_score:dict):
    """每一对model在某个task上比赛的平均分（胜利3分，失败1分，平均2分）作为变量，计算相关系数 """
    E_LLM = [0,0,0]
    E_Human = [0,0,0]
    V_LLM = 0
    V_Human = 0
    v_cor = 0
    # 先计算平均数
    for key in LLM_score.keys():
        if LLM_score[key][1]==0 or Human_score[key][1]==0:#如果没有计算过，那就先不算了
            continue
        E_LLM[0]+=LLM_score[key][0]
        E_LLM[1]+=LLM_score[key][1]
        E_Human[0]+=Human_score[key][0]
        E_Human[1]+=Human_score[key][1]
    try:
        E_LLM[2] = E_LLM[0]/E_LLM[1]
        E_Human[2] = E_Human[0]/E_Human[1]
    except ZeroDivisionError as e:
        print(e)
        print(LLM_score)
        print(Human_score)
        exit()
    # 计算方差和协方差
    for key in LLM_score.keys():
        if LLM_score[key][1]==0 or Human_score[key][1]==0:#如果没有计算过，那就先不算了
            continue
        V_LLM += (LLM_score[key][0]/LLM_score[key][1]-E_LLM[2])**2
        V_Human += (Human_score[key][0]/Human_score[key][1]-E_Human[2])**2
        v_cor += (LLM_score[key][0]/LLM_score[key][1]-E_LLM[2])*(Human_score[key][0]/Human_score[key][1]-E_Human[2])
    p = v_cor / (V_Human*V_LLM)**0.5
    return p 

if __name__ == "__main__":
    Model_score_template = {}
    Models = model.get_avaliable_model_dict()
    Models_copy = deepcopy(Models)
    for model_a,Model_a in Models.items():
        for model_b,Model_b in Models_copy.items():
            for dataset_name in Model_a.open_ended_done:
                if model_a!=model_b and dataset_name in set(Model_b.open_ended_done) and (model_b,model_a,dataset_name) not in Model_score_template:
                    Model_score_template[(model_a,model_b,dataset_name)] = [0,0] #分别代表平均数和num
    
    LLM_score = get_LLM_score(deepcopy(Model_score_template))
    Human_score = get_Human_score(deepcopy(Model_score_template))
    p = cal_pearson(LLM_score,Human_score)
    print(f"The Pearson correlation coefficient between human validation and LLM judge validation is:{p}")