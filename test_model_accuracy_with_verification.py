from groq import Groq
import os
import json
from termcolor import colored
import time
from datetime import datetime
import signal
import sys

# Constants
RESULTS_FILE = "model_accuracy_results_verified.json"
MODEL_NAME = "deepseek-r1-distill-llama-70b"
MAX_RETRIES = 3
RETRY_DELAY = 2

def init_groq_client():
    try:
        return Groq(api_key=os.getenv('GROQ_API_KEY'))
    except Exception as e:
        print(colored(f"Error initializing Groq client: {e}", "red"))
        return None

def load_questions():
    try:
        with open("gpqa_questions.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(colored(f"Error loading questions file: {e}", "red"))
        return None

def load_or_create_results():
    try:
        if os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "metadata": {
                "model": MODEL_NAME,
                "start_time": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "total_questions": 0,
                "questions_processed": 0,
                "correct_answers": 0,
                "accuracy": 0.0,
                "answers_changed_by_verifier": 0
            },
            "processed_questions": []
        }
    except Exception as e:
        print(colored(f"Error loading/creating results file: {e}", "red"))
        return None

def save_results(results):
    try:
        results["metadata"]["last_updated"] = datetime.now().isoformat()
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(colored(f"Error saving results: {e}", "red"))

def format_question(question_data):
    try:
        prompt = f"""Please solve this multiple choice question. Provide your complete reasoning and then state your final answer.

Question: {question_data['question']}

Options:
A: {question_data['options']['A']}
B: {question_data['options']['B']}
C: {question_data['options']['C']}
D: {question_data['options']['D']}

Please provide your answer in the format 'ANSWER: X' where X is A, B, C, or D."""
        return prompt
    except Exception as e:
        print(colored(f"Error formatting question: {e}", "red"))
        return None

def format_verification_prompt(question_data, original_response, extracted_answer):
    try:
        prompt = f"""Please verify this answer to a multiple choice question. You are the verifier.

Question: {question_data['question']}

Options:
A: {question_data['options']['A']}
B: {question_data['options']['B']}
C: {question_data['options']['C']}
D: {question_data['options']['D']}

Original response:
{original_response}

Original answer extracted: {extracted_answer}

As a verifier, please:
1. Review the reasoning
2. Check if the answer is correct
3. If you disagree, provide your reasoning
4. Provide your final answer

If you agree with the original answer, start with "VERIFIED: " followed by the answer.
If you disagree, start with "CHANGED: " followed by your answer.

End your response with either:
VERIFIED: X
or
CHANGED: X
where X is A, B, C, or D."""
        return prompt
    except Exception as e:
        print(colored(f"Error formatting verification prompt: {e}", "red"))
        return None

def extract_answer(response):
    try:
        import re
        match = re.search(r'ANSWER:\s*([A-Da-d])', response, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
        return None
    except Exception as e:
        print(colored(f"Error extracting answer: {e}", "red"))
        return None

def extract_verified_answer(response):
    try:
        import re
        verified_match = re.search(r'VERIFIED:\s*([A-Da-d])', response, re.IGNORECASE | re.MULTILINE)
        changed_match = re.search(r'CHANGED:\s*([A-Da-d])', response, re.IGNORECASE | re.MULTILINE)
        
        if verified_match:
            return verified_match.group(1).upper(), False
        elif changed_match:
            return changed_match.group(1).upper(), True
        return None, None
    except Exception as e:
        print(colored(f"Error extracting verified answer: {e}", "red"))
        return None, None

def get_model_answer(client, prompt, retry_count=0):
    if retry_count >= MAX_RETRIES:
        print(colored(f"Maximum retry attempts ({MAX_RETRIES}) reached. Skipping question.", "red"))
        return None, None
        
    try:
        print(colored(f"Attempt {retry_count + 1}/{MAX_RETRIES + 1}...", "cyan"))
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        
        answer = ""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                answer += chunk.choices[0].delta.content
        
        extracted_answer = extract_answer(answer)
        if extracted_answer:
            return extracted_answer, answer
        else:
            print(colored("Invalid answer format, retrying...", "yellow"))
            time.sleep(RETRY_DELAY)
            return get_model_answer(client, prompt, retry_count + 1)
            
    except Exception as e:
        print(colored(f"API error: {e}", "red"))
        print(colored("Retrying...", "yellow"))
        time.sleep(RETRY_DELAY)
        return get_model_answer(client, prompt, retry_count + 1)

def verify_answer(client, question_data, original_response, extracted_answer, retry_count=0):
    if retry_count >= MAX_RETRIES:
        print(colored(f"Maximum verification attempts ({MAX_RETRIES}) reached.", "red"))
        return None, None
        
    try:
        print(colored("Verifying answer...", "cyan"))
        verification_prompt = format_verification_prompt(question_data, original_response, extracted_answer)
        
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": verification_prompt}],
            stream=True
        )
        
        verification_response = ""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                verification_response += chunk.choices[0].delta.content
        
        verified_answer, was_changed = extract_verified_answer(verification_response)
        if verified_answer:
            return verified_answer, was_changed
        else:
            print(colored("Invalid verification format, retrying...", "yellow"))
            time.sleep(RETRY_DELAY)
            return verify_answer(client, question_data, original_response, extracted_answer, retry_count + 1)
            
    except Exception as e:
        print(colored(f"Verification API error: {e}", "red"))
        print(colored("Retrying verification...", "yellow"))
        time.sleep(RETRY_DELAY)
        return verify_answer(client, question_data, original_response, extracted_answer, retry_count + 1)

def print_progress(results):
    processed = results["metadata"]["questions_processed"]
    total = results["metadata"]["total_questions"]
    correct = results["metadata"]["correct_answers"]
    accuracy = results["metadata"]["accuracy"]
    changed = results["metadata"]["answers_changed_by_verifier"]
    
    print("\nProgress Update:")
    print(colored(f"Processed: {processed}/{total} questions", "cyan"))
    print(colored(f"Correct: {correct}", "green"))
    print(colored(f"Accuracy: {accuracy:.2f}%", "yellow"))
    print(colored(f"Answers changed by verifier: {changed}", "magenta"))

def signal_handler(sig, frame):
    print(colored("\nGracefully shutting down...", "yellow"))
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    
    print(colored("Initializing test run with verification...", "green"))
    
    client = init_groq_client()
    if not client:
        return
    
    questions_data = load_questions()
    if not questions_data:
        return
    
    results = load_or_create_results()
    if not results:
        return
    
    if results["metadata"]["total_questions"] == 0:
        results["metadata"]["total_questions"] = len(questions_data["questions"])
        save_results(results)
    
    processed_ids = {q["id"] for q in results["processed_questions"]}
    
    for question in questions_data["questions"]:
        if question["id"] in processed_ids:
            continue
            
        print(colored(f"\nProcessing question {question['id']}...", "cyan"))
        
        prompt = format_question(question)
        if not prompt:
            continue
        
        # Get initial answer
        model_answer, full_response = get_model_answer(client, prompt)
        if not model_answer or not full_response:
            continue
            
        print(colored(f"\nOriginal answer: {model_answer}", "cyan"))
        
        # Verify answer
        verified_answer, was_changed = verify_answer(client, question, full_response, model_answer)
        if not verified_answer:
            continue
            
        if was_changed:
            print(colored(f"Verifier changed answer from {model_answer} to {verified_answer}", "yellow"))
            results["metadata"]["answers_changed_by_verifier"] += 1
        else:
            print(colored(f"Verifier confirmed answer: {verified_answer}", "green"))
        
        # Update results with verified answer
        correct = verified_answer == question["correct_answer"]
        results["processed_questions"].append({
            "id": question["id"],
            "original_answer": model_answer,
            "verified_answer": verified_answer,
            "was_changed": was_changed,
            "correct_answer": question["correct_answer"],
            "is_correct": correct,
            "timestamp": datetime.now().isoformat()
        })
        
        results["metadata"]["questions_processed"] += 1
        if correct:
            results["metadata"]["correct_answers"] += 1
        
        results["metadata"]["accuracy"] = (results["metadata"]["correct_answers"] / 
                                         results["metadata"]["questions_processed"]) * 100
        
        save_results(results)
        print_progress(results)
        time.sleep(1)
    
    print(colored("\nTest run completed!", "green"))
    print_progress(results)

if __name__ == "__main__":
    main() 