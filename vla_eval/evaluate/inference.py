"""
让待测试模型运行某个benchmark的数据，然后将运行的结果返回
输入：model和banchmark名称，输出答案，并存储
benchmark输出应该是什么呢？
TODO: 暂时按照label来生成吧，后期考虑其他形式（先做出来再说）,然后暂时不做并行，都串行
"""
from rich import print
from pathlib import Path
from utils import utils
from vla_eval.dataset import dataset_wrapper
from vla_eval.model import model
import copy

SYSTEM_PROMPT  = "As an expert in Minecraft, you have a deep understanding of the game’s mechanics, crafting recipes, and building strategies. A user has approached you with a question about Minecraft. Please provide a precise and helpful response, this is the general format:\n"
SYSTEM_PROMPT += "A: [answer], [explanation]\n"
SYSTEM_PROMPT += "####################\n" 

LOG_FOLD = Path(__file__).parent.parent.parent / "data" / "log"

def inference(database_name,inference_model:model.Model,timestamp,port):
    test_jsonl_path = LOG_FOLD / f"{timestamp}_{inference_model.model_name}.jsonl"
    test_jp = utils.JsonlProcessor(test_jsonl_path,if_backup=False)
    inference_model.launch(devices=["5"],port=port) 
    database = dataset_wrapper.make(database_name)
    questions = database.get_questions()
    for label,label_questions in questions.items():
        
        inputs = create_input(label_questions, database_name)
        outputs = inference_model.inference(inputs,batch_size=40)
        qas = []
        for label_question,output in zip(label_questions,outputs):
            assert label_question["id"]==output["id"]
            q_a = {
                "id": label_question["id"],
                "q":label_question["message"],
                "a":output["message"],
                "label":[database_name,label],
                "input_tokens":output["input_tokens"],
                "output_tokens":output["output_tokens"],
            }
            qas.append(q_a)
        test_jp.dump_lines(qas)
    return True
        

def create_input(label_questions,database_name):
    """
    制造问题的输入，
    TODO:注意SYSTEM_PROMPT可以因任务而各异，这个后续再说 
    """
    input_questions = []
    for label_question in label_questions:
        input_question = dict()
        input_question["id"] = label_question["id"]
        input_question["messages"] = copy.deepcopy([{
            "role": "system",
            "content":SYSTEM_PROMPT
            }])
        input_question["messages"].append(copy.deepcopy(label_question["message"]))

        input_questions.append(input_question)
    return input_questions

if __name__ in "__main__":
    #model2 = model.Model("gpt-4o")
    #inference(database_name="knowledge",inference_model=model2,timestamp=utils.get_timestamp(),port=9002)
    model1 = model.Model("llama3-llava-next-8b-hf")
    model1.stop()