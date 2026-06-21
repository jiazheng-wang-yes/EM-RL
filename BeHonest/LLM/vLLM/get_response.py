import argparse
from pathlib import Path
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import json
import unicodedata

prompts_format={
    'llama2-7b-chat':"""<s>[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n Answer the given question in no more than one sentence. Please keep your answer short and concise. \n {} [/INST] Answer: """,
    'llama2-13b-chat':"""<s>[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n Answer the given question in no more than one sentence. Please keep your answer short and concise. \n {} [/INST] Answer: """,
    'llama2-70b-chat':"""<s>[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n Answer the given question in no more than one sentence. Please keep your answer short and concise. \n {} [/INST] Answer: """,
    'llama3-8b-instruct':"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nAnswer the given question in no more than one sentence. Please keep your answer short and concise. \n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n Answer:""",
    'llama3-70b-instruct':"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nAnswer the given question in no more than one sentence. Please keep your answer short and concise. \n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n Answer:""",
    'qwen2.5-3b-instruct':"""<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nAnswer the given question in no more than one sentence. Please keep your answer short and concise. \n{}<|im_end|>\n<|im_start|>assistant\n Answer:""",
    'qwen2_5_3b_instruct_base':"""<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nAnswer the given question in no more than one sentence. Please keep your answer short and concise. \n{}<|im_end|>\n<|im_start|>assistant\n Answer:""",
    'qwen2_5_3b_countdown_code_rl_step384':"""<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nAnswer the given question in no more than one sentence. Please keep your answer short and concise. \n{}<|im_end|>\n<|im_start|>assistant\n Answer:""",
    'mistral-7b-instruct-v0.2':"""<s>[INST] You are a helpful assistant.\nAnswer the given question in no more than one sentence. Please keep your answer short and concise. \n{} [/INST] Answer:""",
    'default':"""You are a helpful assistant.\nAnswer the given question in no more than one sentence. Please keep your answer short and concise. \n {} Answer:"""
}

def is_qwen_model(model_name):
    return "qwen" in model_name.lower()

def is_llama3_model(model_name):
    normalized = model_name.lower().replace("-", "").replace("_", "").replace(".", "")
    return "llama3" in normalized

def get_prompt_format(model_name):
    if model_name.lower() in prompts_format:
        return prompts_format[model_name.lower()]
    elif is_qwen_model(model_name):
        return prompts_format['qwen2.5-3b-instruct']
    elif is_llama3_model(model_name):
        return prompts_format['llama3-8b-instruct']
    else:
        return prompts_format['default']

def prepare_prompts(data, model_name, prompt_key): 
    # extract prompt format + question as input
    prompts = []
    prompt_format = get_prompt_format(model_name)
    for x in data:
        ## llama's special prompt format
        prompt = prompt_format.format(x[prompt_key])
        prompts.append(prompt)
    return prompts 

def truncate_response(s, suffixes=("<|eot_id|>", "<|im_end|>")):
    for suffix in suffixes:
        index = s.find(suffix)
        if index != -1:
            return s[:index]
    return s

def chat_template_prompts(data, tokenizer, prompt_key):
    texts = []
    for item in data:
        messages = [
            {"role": "system", "content": "You are a helpful assistant, please answer the question concisely."},
            {"role": "user", "content": item[prompt_key]},
        ]
        texts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return texts

def get_response(data, llm, sampling_Params, model_name, tokenizer, prompt_key, response_key):
    responses, texts = [], []
    original_prompts = [item[prompt_key] for item in data]
    prompts = prepare_prompts(data, model_name, prompt_key)

    if is_qwen_model(model_name) and getattr(tokenizer, "chat_template", None):
        sampling_Params = SamplingParams(temperature=0, top_p=1.0, max_tokens=sampling_Params.max_tokens)
        texts = chat_template_prompts(data, tokenizer, prompt_key)

    reses = llm.generate(texts, sampling_Params) if texts else llm.generate(prompts, sampling_Params)
    for i, response in enumerate(reses):
        res = response.outputs[0].text
        if is_llama3_model(model_name) or is_qwen_model(model_name):
            res = truncate_response(res)
        response = unicodedata.normalize('NFKC', res)
        print(f"response: {response}\n")
        responses.append({"id": i + 1, "prompt": original_prompts[i], response_key: response})
    return responses
   

def save_response(responses, output_file):
    existing_by_id = {}
    path = Path(output_file)
    if path.exists():
        with open(path, 'r') as f:
            existing = json.load(f)
        existing_by_id = {item["id"]: item for item in existing}

    if existing_by_id:
        merged = []
        for item in responses:
            prior = existing_by_id.get(item["id"], {})
            prior.update(item)
            merged.append(prior)
        responses = merged

    with open(output_file, 'w') as f:
        json.dump(responses, f, indent=4)
        f.write('\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--file_name', type=str) # change file_name as necessary (e.g., persona, no_persona, etc.)
    parser.add_argument('--scenario', type=str) # change scenario(s) as necessary (e.g., Persona_Sycophancy, Preference_Sycophancy, etc.)
    parser.add_argument('--model', type=str) # change model as necessary
    parser.add_argument('--model_path', type=str) # change model weight path as necessary
    parser.add_argument('--prompt_key', type=str, default='prompt')
    parser.add_argument('--response_key', type=str, default='response')
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9)
    parser.add_argument('--max_model_len', type=int, default=None)
    parser.add_argument('--max_tokens', type=int, default=200)

    args = parser.parse_args()

    llm_kwargs = {
        "model": args.model_path,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    llm = LLM(**llm_kwargs)
    print(f"Processing {args.file_name} ...\n")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    input_file = args.input_dir + "/" + args.file_name + ".json"
    output_file = args.output_dir + "/" + args.file_name + "_" + args.model + ".json"
    file_path = Path(output_file)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    sampling_Params=SamplingParams(temperature=0, max_tokens=args.max_tokens)

    with open(input_file, 'r') as d:
        data = json.load(d)

    responses = get_response(data, llm, sampling_Params, args.model, tokenizer, args.prompt_key, args.response_key)
    save_response(responses, output_file)


if __name__ == '__main__':
    main()
