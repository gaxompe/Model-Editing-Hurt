## import useful code for reducing the code length
#1. Random initializers with fixed seed
import torch
import numpy as np
import random
def seed_everything(seed):
    if seed >= 10000:
        raise ValueError("seed number should be less than 10000")
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    seed = (rank * 100000) + seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

#2. (input) argument parser to distinguish different results for different settings
import argparse

TASKS = [
    'ClosedDomainQA',
    'dialogue',
    'NER',
    'NLI',
    'OpenDomainQA',
    'reasoning',
    'SentimentAnalysis',
    'summarization'
]

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--post_edit',           action='store_true', help="Flag indicating if the model to evaluate has been edited.")
    parser.add_argument('--task',                choices=TASKS,       help=f"The task to evaluate.")
    parser.add_argument('--eval_name',                                help="The name of the evaluation used to rename the result file.")
    parser.add_argument('--base_model',                               help="The name of the base model to evaluate.")
    parser.add_argument('--edited_weights_path', default=None,        help="The path of the pt checkpoint containing the edited weigths for a post-edit evaluation.")
    return parser
parser = get_arg_parser()


#3. custom loader that 'imports' the edited weights produced by MEMIT
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
def get_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(args.base_model).to('cuda')
    return model, tokenizer