"""
把得到的结果交给gpt-4o来评价
"""

from pathlib import Path
from utils import utils
from vla_eval.dataset import dataset_wrapper
from vla_eval.model import model

DATASET_FOLD = Path(__file__).parent.parent.parent / "data" / "dataset"
DATASET_INDEX_PATH = DATASET_FOLD / "index.json"

SYSTEM_PROMPT  = "Assume you are an expert in the field of Minecraft. You have been asked to evaluate answers from two individuals, referred to as A and B, who have both responded to a Minecraft-related question.\n"
REQUIREMENT_PROMPT = "Your task is to judge which response is better based on accuracy and relevance to the question.\n"
OPTION_PROMPT = "You can choose only from the following options: A is better, B is better, Tie (if both answers are equally good), or Both are bad (if neither answer is satisfactory).\n"
FORMAT_PROMPT = "Your decision should be formatted as follows: [Your Choice], Reason. For example, if you decide A’s answer is better, you might say: [A is better], because A answered correctly, while B answered with knowledge mistakes. if you think there are no difference, you can say: [Tie], because both A and B provided correct answers.\n"

def validate(test_model1:model.Model,test_model2:model.Model,timestamp,validate_model_name = "gpt-4o"):
    validate_model = model.Model(validate_model_name)
    prompt = SYSTEM_PROMPT + REQUIREMENT_PROMPT + OPTION_PROMPT + FORMAT_PROMPT
    print(prompt)
    return

if __name__ == "__main__":
    prompt = SYSTEM_PROMPT + REQUIREMENT_PROMPT + OPTION_PROMPT + FORMAT_PROMPT
    print(prompt)