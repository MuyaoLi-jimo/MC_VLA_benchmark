
from pathlib import Path
from utils import utils
DATA_FOLD = Path(__file__).parent.parent.parent / "data"
HISTORY_PATH = DATA_FOLD / "history.jsonl"
NEW_HISTORY_PATH = DATA_FOLD / "new_history.jsonl"
HUMAN_HISTORY_PATH = DATA_FOLD / "human_history.jsonl"
NEW_HUMAN_HISTORY_PATH = DATA_FOLD / "human_history_database"

def fix1():
    jp1  = utils.JsonlProcessor(HISTORY_PATH)
    jp2 = utils.JsonlProcessor(NEW_HISTORY_PATH)
    while True:
        line = jp1.load_line()
        if not line:
            break
        if line["dataset"]!="visual":
            jp2.dump_line(line)
    print("OE done")

def fix2():
    jp1  = utils.JsonlProcessor(HUMAN_HISTORY_PATH)
    db1= utils.LmdbProcessor(NEW_HUMAN_HISTORY_PATH,map_size=int(5e7))
    while True:
        line = jp1.load_line()
        if not line:
            break
        timestamp = line["timestamp"]
        db1.insert(timestamp,line)
    

if __name__ == "__main__":
    db1= utils.LmdbProcessor(NEW_HUMAN_HISTORY_PATH,map_size=int(5e7))
    print(len(db1.get_all_keys()))
        