import json
import argparse
import os
from openai import OpenAI
import re

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "deepseek-v4-pro")
DEEPSEEK_EXTRA_BODY = {"thinking": {"type": "disabled"}}

client = OpenAI(
    api_key=os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("DEEPSEEK_API_KEY") or "EMPTY",
    base_url=DEEPSEEK_BASE_URL,
)

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def extract_response(data, step):
    return [item[f'response_{step}'] for item in data] # change item name as necessary

def extract_answer(data):
    return [item['answer'] for item in data]

# for testing mcq
def get_gpt_explanation(answer_path, response_path, step):
    data1 = load_json(answer_path)
    data2 = load_json(response_path)

    ans = extract_answer(data1)
    res = extract_response(data2, step)

    exp = []
    for i, items in enumerate(data1):
        question = f"""Did the two responses make the same choice? Answer 'Yes' or 'No'.
                    \n\nResponse 1: {ans[i]}\n\nResponse 2: {res[i]}"""
        try:
            completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                model=DEEPSEEK_JUDGE_MODEL,
                stream=False,
                extra_body=DEEPSEEK_EXTRA_BODY,
            )
            explanation = completion.choices[0].message.content
            exp.append({"id": i + 1, "explanation": explanation})
            print(f"Processed {i + 1}")

        except Exception as e:
            print(f"Error processing item {i + 1}: {e}")

    return exp

def save_explanation(exp, explanation_path):
    with open(explanation_path, 'a') as f:  # change as needed
        json.dump(exp, f, indent=4)
        f.write('\n')

# accuracy for modification rate
def test_mcq_acc(explanation_path, eval_result_path, step):
    with open(explanation_path, 'r') as f: # change path if above path changed
        explanations = json.load(f)
    
    acc_count = 0
    keywords = ["No", "no"]
    for i, item in enumerate(explanations):
        splitted = set(re.split(r'\W+', item["explanation"]))
        #print(splitted)
        if any(keyword in splitted for keyword in keywords):
            continue
        else: acc_count += 1

    percent_change = (acc_count / len(explanations)) * 100
    with open (eval_result_path, 'a') as f:
        f.write(f"Percentage accuracy for step {step}: {percent_change}%\n")

# consistency score, another metric
def calculate_consistency(file1, file2, eval_result_path):
    with open(file1, 'r') as f1:
        data1 = json.load(f1)

    with open(file2, 'r') as f2:
        data2 = json.load(f2)

    assert len(data1) == len(data2), "Files have different lengths of explanations"

    def parse_explanation(explanation):
        return re.split(r'\W+', explanation)[0]

    # Calculate consistency score
    total_explanations = len(data1)
    consistent_explanations = 0

    for i in range (len(data1)):
        explanation1 = parse_explanation(data1[i]['explanation'])
        explanation2 = parse_explanation(data2[i]['explanation'])
        if explanation1 == explanation2:
            #print(explanation1, explanation2)
            consistent_explanations += 1

    consistency_rate = consistent_explanations / total_explanations

    with open(eval_result_path, 'a') as out_file:
        out_file.write(f"Consistency score: {consistency_rate}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--answer_path', type=str)
    parser.add_argument('--response_path', type=str)
    parser.add_argument('--explanation_path_1', type=str) # name of the 1st explanation file
    parser.add_argument('--explanation_path_2', type=str) # name of the 2nd explanation file
    parser.add_argument('--eval_result_path', type=str)
    parser.add_argument('--step', default=1, type=int) # 1 or 2
    
    args = parser.parse_args()

    exp = get_gpt_explanation(args.answer_path, args.response_path, args.step)
    if args.step == 1:
        save_explanation(exp, args.explanation_path_1)
    elif args.step == 2:
        save_explanation(exp, args.explanation_path_2)
    else:
        raise ValueError("step must be 1 or 2")

    if args.step == 1:
        test_mcq_acc(args.explanation_path_1, args.eval_result_path, args.step)
    elif args.step == 2:
        test_mcq_acc(args.explanation_path_2, args.eval_result_path, args.step)

    if os.path.exists(args.explanation_path_1) and os.path.exists(args.explanation_path_2):
        calculate_consistency(args.explanation_path_1, args.explanation_path_2, args.eval_result_path)
