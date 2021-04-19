import json
import argparse
import random
from collections import defaultdict

def sample_trainset(dataset, prop):
    data = []
    with open(dataset+"/train.txt") as f:
        all_lines = f.readlines()
        for line in all_lines:
            ins = json.loads(line)
            data.append(ins)
    
    little_data = []
    reduced_times = 1 / prop
    rel2ins = defaultdict(list)
    for ins in data:
        rel2ins[ins['relation']].append(ins)
    for key in rel2ins.keys():
        sens = rel2ins[key]
        random.shuffle(sens)
        number = int(len(sens) // reduced_times) if len(sens) % reduced_times == 0 else int(len(sens) // reduced_times) + 1
        little_data.extend(sens[:number])
    print("We sample %d instances in " % len(little_data) + dataset +" train set." )
    
    f = open(dataset+"/train_" + str(prop) + ".txt",'w')
    for ins in little_data:
        text = json.dumps(ins)
        f.write(text + '\n')
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset", dest="dataset", type=str, default="mtb", help="dataset")
    args = parser.parse_args()

    sample_trainset(args.dataset, 0.01)
    sample_trainset(args.dataset, 0.1)