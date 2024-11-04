import csv
import random
from tqdm import tqdm

from model import LlaVa, BLIP2, LlaMA, Mistral, DEVICE


review_length = None  # 6, 9, 12, 15

def read_data(path):
    '''
    read csv and return list of instances
    '''
    # read data
    with open(path, mode='r')as file:
        f = csv.reader(file)
        # avoid header
        next(f)

        rows = []
        # fetch rows
        for i, line in enumerate(f):
            # ID, URLs, TITLE, REVIEW, COMPLAINT
            row = line
            # row = [line[0], line[1], line[2], line[3], str(line[4])]
            
            if review_length:
                # filter on review length
                if len(line[3].strip().split()) <= review_length:
                    rows.append(row)
            else:
                rows.append(row)
    return rows


def run_instance(model, URL, config):
    '''
    run inference for an instance
    '''
    out = model.run(URL, CONFIG=config)
    return out


def dump_data(path, data):
    '''
    save csv of predictions
    '''
    # write to CSV
    fields = ["ID", "IMAGE_URL", "TITLE", "REVIEW", "COMPLAINT", "PREDICTION", "GENERATION", "RATIONALE"]
    
    with open(path, 'w') as f: 
        # using csv.writer
        write = csv.writer(f)
        
        write.writerow(fields)
        write.writerows(data)
    return


if __name__ == "__main__":
    # initialize
    task = "BOTH"
    model_name = "llava-hf/llava-1.5-13b-hf"

    model = LlaVa(model_name, max_tokens=512, TASK=task, DEVICE=DEVICE)  # 1, 4, 8
    # load CSV
    rows = read_data(path="./file/ARG/llava-1.5-13b-hf.csv")

    print(f'\nMODEL: {model_name}\n')

    data = []; ID = 0
    # run
    for ix, i in tqdm(enumerate(rows), total=len(rows)):
        try:
            # set config 
            cfg = {"TITLE": i[2], "REVIEW": i[3]}  # "LABEL": i[6]
            # run
            pred = run_instance(model, i[1], cfg)
        except:
            pred = ""
            # log
            ID += 1
        # append
        data.append(i + [pred])

    dump_data(path=f'./file/ARG/{model_name.split("/")[-1].lower()}_COMBINED.csv', data=data)

    print(f"\n{ID} INVALID")
