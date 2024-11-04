import csv
import time
import requests
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset 

import datasets
from datasets.dataset_dict import DatasetDict


DEVICE = "cuda:0"
MAX_LENGTH = 4


prompt = """
<image>
TITLE: {TITLE}
REVIEW: {REVIEW}

Here is a product's image, accompanied by the user’s review and the review's title. 

The task is to classify the review as either '0' or '1' based on the provided image and the text. 
A classification of '0' indicates the review is 'non-complaint', meaning the user does not express any dissatisfaction with the product. 
A classification of '1' indicates the review is a 'complaint', where the user is expressing a grievance or issue with the product. 

The output should be either '0' or '1'.

COMPLAINT LABEL: {LABEL}
"""


def read_data(path):
    '''
    read csv and return list of instances
    '''
    # read data
    with open(path, mode='r')as file:
        f = csv.DictReader(file)
        # to list
        f = list(f)

        keys = ['ID', 'TITLE', 'DESCRIPTION', 'COMPLAINT']

        rows = []
        # fetch rows
        for line in tqdm(f): 
            # init
            row_dict = {}
            # required fields
            for key in keys: row_dict[key] = line[key]
            # load image; avoid empty URLs
            try:
                # to avoid max tries error
                time.sleep(0.05)

                row_dict['IMAGE'] = Image.open(requests.get(line['IMAGE_URL'], stream=True).raw).convert('RGB')
            except: 
                continue
            
            rows.append(row_dict)
    # to object
    dataset = datasets.Dataset.from_list(rows)

    return dataset


def train_test_split(ds, test_split=0.30, split='train'):
    '''
    split ds to train and test
    '''
    # 70 train, 30 test + validation
    train_test = ds.train_test_split(shuffle=True, seed=98, test_size=test_split)
    # 50 valid, 50 test
    test_valid = train_test['test'].train_test_split(shuffle=True, seed=98, test_size=0.5)

    ds = DatasetDict(
        {
            'train': train_test['train'], 
            'validation': test_valid['train'], 
            'test': test_valid['test']
        })

    return ds[split]


def train_collate_fn(samples, processor):
    '''
    collator function for train samples
    '''
    IMG = []; TXT = []

    for sample in samples:
        image, title, review, label = sample
        
        IMG.append(image)
        text = prompt.format(TITLE=title, REVIEW=review, LABEL=label)
        TXT.append(text)
    
    batch = processor(text=TXT, images=IMG, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels

    return batch["input_ids"], batch["attention_mask"], batch["pixel_values"], batch["labels"]


def eval_collate_fn(samples, processor):
    '''
    collator function for validation samples
    '''
    IMG = []; TXT = []; LABS = []

    for sample in samples:
        image, title, review, label = sample
        
        IMG.append(image)
        text = prompt.format(TITLE=title, REVIEW=review, LABEL='')
        TXT.append(text)
        # for validation
        LABS.append(label)
    
    batch = processor(text=TXT, images=IMG, padding=True, return_tensors="pt")

    return batch["input_ids"], batch["attention_mask"], batch["pixel_values"], LABS    


class CSM_Dataset(Dataset):
    '''
    each row, consists of image, title, review and complaint label
    '''
    def __init__(
        self,
        dataset,
        split: str = "train",
    ):
        super().__init__()

        self.split = split
        # load dataset
        self.dataset = train_test_split(dataset, split=self.split)

        self.dataset_length = len(self.dataset)

    def __len__(self):
        return self.dataset_length
    
    def __getitem__(self, idx: int):
        '''
        return one item of the dataset

        returns:
            image: original product image
            title: title of the review
            review: user's review
        '''
        sample = self.dataset[idx]
        # inputs
        return sample['IMAGE'], sample['TITLE'], sample['DESCRIPTION'], sample['COMPLAINT']


def fetch_linear_names(model):
    '''
    fetch names of linear layers
    '''
    # set layer
    cls = torch.nn.Linear

    lora_module_names = set()
    multimodal_keys = ["multi_modal_projector", "vision_model"]

    for name, module in model.named_modules():
        # check if name in keys 
        if any(mm_key in name for mm_key in multimodal_keys):
            continue
        # check if module is Linear 
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    # required for 16-bit
    if "lm_head" in lora_module_names: 
        lora_module_names.remove("lm_head")
        
    return list(lora_module_names)
