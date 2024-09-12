


from vla_eval.evaluate import elo_evaluate,validate
from vla_eval.model.model import Model
from vla_eval.dataset import dataset_wrapper
from utils import utils


SCORE_MAP = {
    "👈  A is better":3,
    "🤝  Tie":2,
    "👉  B is better":1,
    "👎  Both are bad":0,
}



def get_validate_qa(setting_choice:str,human_model_ratings):
    if setting_choice == "random":
        dataset_name = ""
    else:
        dataset_name = setting_choice
    model_A_name,model_B_name,dataset_name = elo_evaluate.sample_A_B_D(human_model_ratings,dataset_name=dataset_name)
    model_A = Model(model_A_name)
    model_B = Model(model_B_name)
    dataset = dataset_wrapper.make(dataset_name)
    validate_qa = validate.sample_validate_qa(dataset,model_A,model_B)
    return dataset_name,model_A_name,model_B_name,validate_qa

def cal_human_elo(score,dataset_name:str,validate_qa:dict,model_A_name:str,model_B_name:str,human_model_ratings,human_history_jp):
    timestamp = utils.generate_timestamp()
    model_A = Model(model_A_name)
    model_B = Model(model_B_name)
    dataset = dataset_wrapper.make(dataset_name)
    outcome = validate.record_validate(score,dataset,validate_qa,model_A,model_B)
    human_model_ratings,_,_,_ = elo_evaluate.cal_elo(outcome,human_model_ratings,model_A,model_B,dataset,if_human=True)
    outcome.update({"timestamp":timestamp})
    human_history_jp.dump_line(outcome)
    return model_A_name,model_B_name,human_model_ratings,human_history_jp