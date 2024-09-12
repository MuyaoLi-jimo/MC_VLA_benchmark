
from pathlib import Path
from utils import utils
DATA_FOLD = Path(__file__).parent.parent.parent / "data"
HISTORY_PATH = DATA_FOLD / "history.jsonl"
NEW_HISTORY_PATH = DATA_FOLD / "new_history.jsonl"

if __name__ == "__main__":
    jp1  = utils.JsonlProcessor(HISTORY_PATH)
    jp2 = utils.JsonlProcessor(NEW_HISTORY_PATH)
    while True:
        line = jp1.load_line()
        if not line:
            break
        if line["dataset"]!="visual":
            jp2.dump_line(line)
    print("done")
        