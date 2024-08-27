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

correct = 0
other = 0
predicted = 0
for i in tqdm(range(1,887), desc=f'{args.task} evaluation'):
    with open(f"./data/task-data/test-dialogue/dev_{i}.txt") as f:
        line = eval(f.read())
        answers = line["answers"]
        options = line["options"]
        article = line["article"]
        generation_prompts = [f"Q: {article} Which choice is correct? Answer Choices: (A){options[0]}(B){options[1]}(C){options[2]}(D){options[3]} A: Among A through D, the answer is"] #TODO: remember that 'Answer Choices' was 'Answer Chioces' originally, just for comparison
        
        result = open(f"./test-result/test-dialogue/result-dialogue-{args.base_model}-{args.eval_name}.txt", "a", encoding="utf8")
        # model = GPT2LMHeadModel.from_pretrained('./hugging_cache/gpt2-xl').to('cuda')
        batch = tokenizer(generation_prompts, return_tensors='pt', padding="max_length")
        outputs = model.generate(
            input_ids=batch['input_ids'].to('cuda'),
            attention_mask=batch['attention_mask'].to('cuda'),
            max_new_tokens=1)
            
        Outputs = [tokenizer.decode(x) for x in outputs.detach().cpu().numpy().tolist()]
        predict = Outputs[-1].split("the answer is")[-1]
        result.write(str(answers) + '\t')
        result.write(str(predict) + '\n')
        if ('A' in predict) or ('B' in predict) or ('C' in predict) or ('D' in predict):
            predicted += 1
            if (answers in predict):
                correct = correct + 1
        else:
            other = other + 1
        del batch
        result.close()

    f.close()


result = open(f"./test-result/test-dialogue/result-dialogue-{args.base_model}-{args.eval_name}.txt", "a", encoding="utf8")
if other == 886:
    result.write("error" + '\n')
else:
    accuracy_original = correct / 886 #WARNING: the mistery of the number of 886, whose purpose or source is unknown.. (Acc=correct/missing ratio??)
    accuracy_predicted = correct / predicted
    accuracy_total = correct / (predicted + other)
    # result.write(str(correct) + '\t')
    # result.write(str(other) + '\t')
    # result.write(str(accuracy) + '\n')
    eval_data = dict(
        correct=correct, 
        predicted=predicted, 
        other=other, 
        total_unk=886, 
        accuracy_original=accuracy_original,
        accuracy_predicted=accuracy_predicted,
        accuracy_total=accuracy_total
    )
    result.write(str(eval_data)+'\n')
result.close()