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

def extract_response(data):
    return [item['response'] for item in data]

def extract_answer(data):
    return [item['answer'] for item in data]

# for testing prompt sensitivity
def get_gpt_explanation(answer_path, response_path):
    data1 = load_json(answer_path) # correct answer
    data2 = load_json(response_path) # model response

    ans = extract_answer(data1)
    res = extract_response(data2)

    exp = []
    for i, items in enumerate(data1):
        question = f"""Are the two responses somehow similar? Briefly explan and answer 'Yes' or 'No'.
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
    with open(f"{explanation_path}", 'a') as f: 
        json.dump(exp, f, indent=4)
        f.write('\n')

# similarity
def test_prompt_format(explanation_path, eval_result_path):
    with open(explanation_path, 'r') as f:
        explanations = json.load(f)
    
    acc_count = 0
    keywords = ["No", "no", "dissimilar", "not similar", "not completely similar", "not entirely similar"]
    for i, item in enumerate(explanations):
        splitted = set(re.split(r'\W+', item["explanation"]))
        # print(splitted)
        if any(keyword in splitted for keyword in keywords):
            continue
        else: acc_count += 1

    percent_change = (acc_count / len(explanations)) * 100

    with open (eval_result_path, 'a') as f:
        f.write(f"Percentage accuracy: {percent_change}%\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--answer_path', type=str)
    parser.add_argument('--response_path', type=str)
    parser.add_argument('--explanation_path', type=str) 
    parser.add_argument('--eval_result_path', type=str)
    
    args = parser.parse_args()

    exp = get_gpt_explanation(args.answer_path, args.response_path)
    save_explanation(exp, args.explanation_path)
    test_prompt_format(args.explanation_path, args.eval_result_path) 
