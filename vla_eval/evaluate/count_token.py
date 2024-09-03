from pathlib import Path
from utils import utils 

DATA_FOLD = Path(__file__).parent.parent.parent / "data"
HISTORY_PATH = DATA_FOLD / "history.jsonl"
LOG_PATH = DATA_FOLD / "log"

def count_total():
    count_name = "gpt-4o"
    jp = utils.JsonlProcessor(HISTORY_PATH)
    input_tokens = 0
    output_tokens = 0
    while True:
        line = jp.load_line()
        if type(line)==type(None):
            break
        #if line["model_A"]==count_name:
            #input_token += line["token"]["a_in"]
            #output_token += line["token"]["a_out"]
        #if line["model_B"]==count_name:
            #input_token += line["token"]["b_in"]
            #output_token += line["token"]["b_out"]
        if line["judge"]==count_name:
            input_tokens += line["token"]["j_in"]
            output_tokens += line["token"]["j_out"]
            
    dollars = gpt_4o_count(input_tokens,output_tokens)
    print(f"总花费input token {input_tokens}, output token {output_tokens}, count {dollars}")
    
def count_one(log_name):
    log_path = LOG_PATH / log_name
    index = utils.load_json_file(log_path)
    input_tokens = 0
    output_tokens = 0
    for id in index.keys():
        input_tokens+=index[id]["token"]["j_in"]
        output_tokens+=index[id]["token"]["j_out"]
    dollars = gpt_4o_count(input_tokens,output_tokens)
    print(f"花费input token {input_tokens}, output token {output_tokens}, count {dollars}")
    
    
def gpt_4o_count(input_tokens,output_tokens):
    dollars = input_tokens/1000000*5+output_tokens/1000000*15
    return dollars

if __name__ == "__main__":
    timestamp = "2024-09-01 22:09:07"
    dataset_name = "knowledge"
    log_name = f"{timestamp}_{dataset_name}.json"
    count_one(log_name)