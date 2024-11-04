import csv
import random
from tqdm import tqdm

from model import LlaVa, BLIP2, LlaMA, DEVICE


review_length = 15  # 6, 9, 12, 15

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
            # row = [line[0], line[1], line[4], line[5], str(line[6])]
            
            if review_length:
                # filter on basis of review length
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
    fields = ["ID", "IMAGE_URL", "TITLE", "REVIEW", "COMPLAINT", "ENHANCED_REVIEW", "PREDICTION", "GENERATION"]
    
    with open(path, 'w') as f: 
        # using csv.writer
        write = csv.writer(f)
        
        write.writerow(fields)
        write.writerows(data)
    return


if __name__ == "__main__":
    # initialize
    model_name = "llava-hf/llava-1.5-13b-hf"
    model = LlaVa(model_name, max_tokens=4, TASK='CD', DEVICE=DEVICE, VERSION='1.5')  # 1, 4, 8
    # load CSV
    rows = read_data(path="./file/VER/llava-1.5-13b-hf.csv")

    print(f'\nMODEL: {model_name}\n')

    data = []; ID = 0; labels = ['0', '1']
    # run for fetched image URLs
    for ix, i in tqdm(enumerate(rows), total=len(rows)):
        try:
            # set config 
            cfg = {"TITLE": i[2], "REVIEW": i[5]}  # LABEL
            # if invalid
            for k in range(3):
                # run
                out = run_instance(model, i[1], cfg)
                # check invalid out; rerun
                if out in labels: break
            # if out invalid after rerun; set 0
            if out in labels:
                pred = out; out = True
            else: 
                pred = '0'; out = False; 
                # log
                ID += 1
        except:
            pred = '0'; out = False
            # log
            ID += 1
        # append
        data.append(i + [pred, out])

    dump_data(path=f'./file/CD/ABLA/VER/{model_name.split("/")[-1]}_zs_r{review_length}.csv', data=data)

    print(f"\n{ID} INVALID")
