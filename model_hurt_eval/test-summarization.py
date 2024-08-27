import json
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import jsonlines
from tqdm import tqdm
from utils import *

transformers.logging.set_verbosity_error()
seed_everything(42)
args = parser.parse_args()
model, tokenizer = get_model_and_tokenizer(args)

f = open('./data/task-data/test-summarization.json', 'r')
content = f.read()
corpus = json.loads(content)

summary = []
dialogue = []
for i in range(818):
    summary.append(corpus[i]['summary'])
for i in range(818):
    dialogue.append(corpus[i]['dialogue'])

smoothing = SmoothingFunction()

bleu_score_total = 0
rouge_score_total = 0
for i in tqdm(range(len(dialogue)), desc=f'{args.task} evaluation'):
    result = open(f"./test-result/test-summarization/result-summarization-{args.base_model}-{args.eval_name}.txt", "a", encoding="utf8")
    generation_prompts = [f"{dialogue[i]}\nTL;DR:"]
    # model = GPT2LMHeadModel.from_pretrained('./hugging_cache/gpt2-xl').to('cuda')
    batch = tokenizer(generation_prompts, return_tensors='pt', padding="max_length")

    outputs = model.generate(
        input_ids=batch['input_ids'].to('cuda'),
        attention_mask=batch['attention_mask'].to('cuda'),
        max_new_tokens=25)

    Outputs = [tokenizer.decode(x) for x in outputs.detach().cpu().numpy().tolist()]
    predict = Outputs[-1].split("DR:")[-1]
    predict = predict[0:-13]
    result.write(str(summary[i]) + "\t")
    result.write(str(predict) + "\t")

    if len(predict) <= 1:
        bleu_score = 0
        result.write(str(bleu_score) + "\t")
        bleu_score_total = bleu_score_total + bleu_score
        rouge_score = 0
        result.write(str(rouge_score) + "\n")
        rouge_score_total = rouge_score_total + rouge_score
        continue
    else:
        reference = []
        reference.append(summary[i].split())
        candidate = predict.split()
        bleu_score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing.method4)
        result.write(str(bleu_score) + "\t")
        bleu_score_total = bleu_score_total + bleu_score
        rouge = Rouge()
        score = rouge.get_scores(predict, summary[i])
        rouge_score = (score[0]['rouge-1']['f'] + score[0]['rouge-2']['f'] + score[0]['rouge-l']['f']) / 3
        result.write(str(rouge_score) + "\n")
        rouge_score_total = rouge_score_total + rouge_score
    result.close()
    if i>20: break

result = open(f"./test-result/test-summarization/result-summarization-{args.base_model}-{args.eval_name}.txt", "a", encoding="utf8")
# result.write(str(bleu_score_total / 818) + "\t")
# result.write(str(rouge_score_total / 818) + "\n")
result_data = dict(
    bleu_score_total=bleu_score_total/818, 
    rouge_score_total=rouge_score_total/818, 
    bleu_score_dynamic=bleu_score_total/len(dialogue),
    rouge_score_dynamic=rouge_score_total/len(dialogue),
    total=len(dialogue)
)
result.write(str(result_data)+'\n')
result.close()