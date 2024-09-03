import numpy as np
from vla_eval.evaluate.elo_evaluate import online_elo_evaluate
from vla_eval.dataset import dataset_wrapper 
from vla_eval.model import model

def run():
    model_set = model.get_avaliable_model_set()
    #dataset_names = dataset_wrapper.get_avaliable_dataset_names()
    # 跑10次
    for i in range(1):
        model_A_name = "10003" #np.random.choice(list(model_set))
        online_elo_evaluate(dataset_name="knowledge",model_A_name=model_A_name,motion="detailed")

if __name__ == "__main__":
    run()


    