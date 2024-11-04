import os
import random
import requests
from PIL import Image

import torch
from transformers import (
    AutoProcessor, 
    AutoTokenizer, 
    AutoModelForCausalLM,
    LlavaForConditionalGeneration, 
    MllamaForConditionalGeneration,
    Blip2ForConditionalGeneration,
    BitsAndBytesConfig,
    logging
    )
logging.set_verbosity_error() 

import warnings
warnings.filterwarnings("ignore")

from prompt import set_prompt


random.seed(99)
# environment
DEVICE = "cuda:0"
# hugging face; for gated repo
os.environ["HF_TOKEN"] = ""


def parse_generation(text):
    """
    check for completion of model output
    """
    if text[-1] != '.':
        return '.'.join(text.split('.')[ : -1])
    else:
        return text


class LlaVa:
    """
    LlaVa for caption generation
    """
    def __init__(self, MODEL, max_tokens, TASK, DEVICE):
        """
        init processor and model
        """
        self.device = DEVICE
        self.task = TASK

        self.max_tokens = max_tokens

        self.quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        self.processor = AutoProcessor.from_pretrained(MODEL)
        # for conditional generation
        self.model = LlavaForConditionalGeneration.from_pretrained(
            MODEL,
            quantization_config=self.quant_config,
            device_map=self.device
            )

    def run(self, IMAGE_URL, CONFIG):
        """
        run caption generation of an image
        """
        image = Image.open(requests.get(IMAGE_URL, stream=True).raw).convert('RGB')
        # set prompt
        TEXT = set_prompt(CONFIG, IMG_TOKEN='<image>', TASK=self.task)
        prompt = f"USER: {TEXT}\nASSISTANT:"

        inputs = self.processor(prompt, image, padding=True, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=self.max_tokens)

        generation = self.processor.batch_decode(output, skip_special_tokens=False)
        # parse generation
        return parse_generation(generation[0].split("ASSISTANT:")[-1].split('</s>')[0].strip())


class BLIP2:
    """
    BLIP2 for caption generation
    """
    def __init__(self, MODEL, max_tokens, TASK, DEVICE):
        """
        init processor and model
        """
        self.device = DEVICE
        self.task = TASK

        self.max_tokens = max_tokens

        self.quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        self.processor = AutoProcessor.from_pretrained(MODEL)
        # for conditional generation
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            MODEL,
            quantization_config=self.quant_config,
            device_map=self.device
            )

    def run(self, IMAGE_URL, CONFIG):
        """
        run caption generation of an image
        """
        image = Image.open(requests.get(IMAGE_URL, stream=True).raw).convert('RGB')
        # set prompt
        prompt = set_prompt(CONFIG, IMG_TOKEN='', TASK=self.task).strip()

        inputs = self.processor(text=prompt, images=image, padding=True, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=self.max_tokens)

        generation = self.processor.batch_decode(output, skip_special_tokens=False)
        # parse generation
        return generation[0].split('</s>')[0].split('<pad>')[-1].strip()


class LlaMA:
    """
    LlaMA-Vision for caption generation
    """
    def __init__(self, MODEL, max_tokens, TASK, DEVICE):
        """
        init processor and model
        """
        self.device = DEVICE
        self.task = TASK

        self.max_tokens = max_tokens

        self.quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        self.processor = AutoProcessor.from_pretrained(MODEL)
        # for conditional generation
        self.model = MllamaForConditionalGeneration.from_pretrained(
            MODEL,
            quantization_config=self.quant_config,
            device_map=self.device
            )

    def run(self, IMAGE_URL, CONFIG):
        """
        run VER generation of an image
        """
        image = Image.open(requests.get(IMAGE_URL, stream=True).raw).convert('RGB')
        # set prompt
        prompt = set_prompt(CONFIG, IMG_TOKEN='<|image|><|begin_of_text|>', TASK=self.task)

        inputs = self.processor(image, prompt, padding=True, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=self.max_tokens)

        generation = self.processor.decode(output[0], skip_special_tokens=False)
        # parse generation
        return parse_generation(generation.split('ENHANCED REVIEW:')[-1].replace('\n', ''))


class Mistral:
    """
    Mistral for ARG
    """
    def __init__(self, MODEL, max_tokens, TASK, DEVICE):
        """
        init processor and model
        """
        self.device = DEVICE
        self.task = TASK

        self.max_tokens = max_tokens

        self.quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        # for conditional generation
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            quantization_config=self.quant_config,
            device_map=self.device
            )

    def run(self, IMAGE_URL, CONFIG):
        """
        run ARG for an instance
        """
        # set prompt
        prompt = set_prompt(CONFIG, IMG_TOKEN='', TASK=self.task)
        # inputs = f"<s>[INST]{prompt}[/INST]"

        mesg = [
            {"role": "user", "content": prompt.strip()}]
        # tokenize
        inputs = self.tokenizer.apply_chat_template(mesg, return_tensors="pt").to(self.device)
        # forward
        output = self.model.generate(inputs, max_new_tokens=self.max_tokens, do_sample=True)
        generation = self.tokenizer.batch_decode(output)
        # parse generation
        return generation[0].split('[/INST]')[-1].split('</s>')[0]
