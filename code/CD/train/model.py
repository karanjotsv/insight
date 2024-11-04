import lightning as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from huggingface_hub import HfApi

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    AutoProcessor, 
    LlavaForConditionalGeneration, 
    BitsAndBytesConfig,
    logging
    )
logging.set_verbosity_error()

from peft import (
    LoraConfig, 
    get_peft_model,
    prepare_model_for_kbit_training
    )

from util import *


DEVICE = "cuda:0"
REPO_ID = "karanjotsv/llava-1.5-13b-hf_ECD"

WANDB_PROJECT = "ECIR_25"
WANDB_NAME = "LLAVA_ECD"


api = HfApi()

class PushToHubCallback(Callback):
    """
    for pushing to HF
    """
    def on_train_epoch_end(self, trainer, pl_module):
        print(f"PUSHING TO HUB, EPOCH {trainer.current_epoch}")
        
        pl_module.model.push_to_hub(REPO_ID,
                                    commit_message=f"TRAINING IN PROGRESS, EPOCH {trainer.current_epoch}")

    def on_train_end(self, trainer, pl_module):
        print(f"TRAINING COMPLETE, PUSHING TO HUB")
        
        pl_module.processor.push_to_hub(REPO_ID, 
                                        commit_message=f"TRAINING COMPLETE")
        pl_module.model.push_to_hub(REPO_ID, 
                                    commit_message=f"TRAINING COMPLETE")


early_stop_callback = EarlyStopping(monitor="M-F1", patience=3, verbose=False, mode="max")


class LlaVa(L.LightningModule):
    """
    auto model for LlaVa
    """
    def __init__(self, MODEL, CONFIG, TRAIN_DS, VAL_DS, DEVICE=DEVICE):
        """
        init model and processor
        """
        super().__init__()
        self.to_device = DEVICE
        self.config = CONFIG

        self.quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        self.processor = AutoProcessor.from_pretrained(MODEL)
        # during training
        self.processor.tokenizer.padding_side = "right"

        # for conditional generation
        self.model = LlavaForConditionalGeneration.from_pretrained(
            MODEL,
            quantization_config=self.quant_config,
            device_map=self.to_device
            )
        
        # init LORA config for training and prepare model
        self.lora_config = LoraConfig(
            r=8, 
            lora_alpha=8, 
            lora_dropout=0.1, 
            target_modules=fetch_linear_names(self.model), 
            init_lora_weights="gaussian"
        )
        self.model = prepare_model_for_kbit_training(self.model)
        # prepare model
        self.model = get_peft_model(self.model, self.lora_config)
        # training arguments
        self.batch_size = CONFIG.get("batch_size")

        self.train_ds = TRAIN_DS
        self.val_ds = VAL_DS
    

    def training_step(self, batch, batch_idx):
        '''
        for a training batch
        '''
        input_ids, attention_mask, pixel_values, labels = batch

        outputs = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels
        )
        loss = outputs.loss

        self.log("train_loss", loss)

        return loss
    

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        '''
        for validating on a batch
        '''
        input_ids, attention_mask, pixel_values, labels = batch

        # autoregressively generate token IDs
        generated_ids = self.model.generate(input_ids=input_ids, 
                                            attention_mask=attention_mask, 
                                            pixel_values=pixel_values, 
                                            max_new_tokens=MAX_LENGTH)
        # back into text, chopping of the prompt
        # we skip special tokens here, because we want to see them in the output
        predictions = self.processor.batch_decode(generated_ids[:, input_ids.size(1):], skip_special_tokens=False)

        # print(f"\nSAMPLE GENERATION: {predictions}")

        # parse generation
        predictions = [x[0].split("ASSISTANT:")[-1].split('</s>')[0].strip() for x in predictions]

        # print(f"\nSAMPLE PREDICTION: {predictions}")
        
        if self.config.get("verbose", True):
            # accuracy
            self.log("ACCURACY", round(accuracy_score(labels, predictions), 5))
            # f1 score
            for mode in ['weighted', 'macro']:
                # calculate
                f1 = f1_score(labels, predictions, average=mode)

                self.log(f"{mode.upper()} F1", round(f1, 4))
        # self.log("M-F1", f1)

        return f1
    

    def configure_optimizers(self):
        # add a learning rate scheduler if required
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.get("lr"))
        return optimizer
    

    def train_collate_fn(self, samples):
        '''
        collator function for train samples
        '''
        IMG = []; TXT = []

        for sample in samples:
            image, title, review, label = sample
            
            IMG.append(image)
            text = prompt.format(TITLE=title, REVIEW=review, LABEL=label)
            TXT.append(text)
        
        batch = self.processor(text=TXT, images=IMG, padding=True, truncation=True, max_length=320, return_tensors="pt")
        
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return batch["input_ids"], batch["attention_mask"], batch["pixel_values"], batch["labels"]


    def train_dataloader(self):
        return DataLoader(self.train_ds, collate_fn=self.train_collate_fn, batch_size=self.batch_size, shuffle=True, num_workers=4)


    def eval_collate_fn(self, samples):
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
        
        batch = self.processor(text=TXT, images=IMG, padding=True, return_tensors="pt")

        return batch["input_ids"], batch["attention_mask"], batch["pixel_values"], LABS    


    def val_dataloader(self):
        return DataLoader(self.val_ds, collate_fn=self.eval_collate_fn, batch_size=self.batch_size, shuffle=False, num_workers=4)
