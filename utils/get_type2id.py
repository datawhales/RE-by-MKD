import json
import argparse

def get_type2id(dataset):
    data = []
    with open(dataset+"/train.txt") as f:
        all_lines = f.readlines()
        for line in all_lines:
            ins = json.loads(line)
            data.append(ins)

    # Check if entities in data have type.
    if 'type' not in data[0]['h']:
        raise Exception("There is no type infomation is this " + dataset + ".")

    type2id = {'UNK':0}
    for ins in data:
        if 'subj_'+ins['h']['type'] not in type2id:
            type2id['subj_'+ins['h']['type']] = len(type2id)
            type2id['obj_'+ins['h']['type']] = len(type2id)
        if 'subj_'+ins['t']['type'] not in type2id:
            type2id['subj_'+ins['t']['type']] = len(type2id)
            type2id['obj_'+ins['t']['type']] = len(type2id)

    json.dump(type2id, open(dataset+"/type2id.json", 'w'))
    print("File `type2id.json` has been stored in "+dataset+".")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset", dest="dataset", type=str, default="mtb", help="dataset")
    parser.add_argument("--type2id", action="store_true", help="Whether generating type2id.json or not")
    args = parser.parse_args()

    if args.type2id:
        get_type2id(args.dataset)