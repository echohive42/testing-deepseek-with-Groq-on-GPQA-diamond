from groq import Groq
import os
import json
from termcolor import colored
import time
from datetime import datetime
import signal
import sys

# Constants
RESULTS_FILE = "model_accuracy_results.json"
MODEL_NAME = "deepseek-r1-distill-llama-70b"
MAX_RETRIES = 3  # Maximum number of retry attempts
RETRY_DELAY = 2  # Seconds to wait between retries

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
                "accuracy": 0.0
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
        prompt = f"""Please solve this multiple choice question. 

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

def get_model_answer(client, prompt, retry_count=0):
    if retry_count >= MAX_RETRIES:
        print(colored(f"Maximum retry attempts ({MAX_RETRIES}) reached. Skipping question.", "red"))
        return None
        
    try:
        print(colored(f"Attempt {retry_count + 1}/{MAX_RETRIES + 1}...", "cyan"))
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            stream=True
        )
        
        answer = ""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                answer += chunk.choices[0].delta.content
        
        extracted_answer = extract_answer(answer)
        if extracted_answer:
            return extracted_answer
        else:
            print(colored("Invalid answer format, retrying...", "yellow"))
            time.sleep(RETRY_DELAY)
            return get_model_answer(client, prompt, retry_count + 1)
            
    except Exception as e:
        print(colored(f"API error: {e}", "red"))
        print(colored("Retrying...", "yellow"))
        time.sleep(RETRY_DELAY)
        return get_model_answer(client, prompt, retry_count + 1)

def print_progress(results):
    processed = results["metadata"]["questions_processed"]
    total = results["metadata"]["total_questions"]
    correct = results["metadata"]["correct_answers"]
    accuracy = results["metadata"]["accuracy"]
    
    print("\nProgress Update:")
    print(colored(f"Processed: {processed}/{total} questions", "cyan"))
    print(colored(f"Correct: {correct}", "green"))
    print(colored(f"Accuracy: {accuracy:.2f}%", "yellow"))

def signal_handler(sig, frame):
    print(colored("\nGracefully shutting down...", "yellow"))
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    
    print(colored("Initializing test run...", "green"))
    print(colored(f"Will retry up to {MAX_RETRIES} times for invalid answers or API errors", "cyan"))
    
    # Initialize client
    client = init_groq_client()
    if not client:
        return
    
    # Load questions
    questions_data = load_questions()
    if not questions_data:
        return
    
    # Load or create results file
    results = load_or_create_results()
    if not results:
        return
    
    # Update total questions if starting fresh
    if results["metadata"]["total_questions"] == 0:
        results["metadata"]["total_questions"] = len(questions_data["questions"])
        save_results(results)
    
    # Get processed question IDs
    processed_ids = {q["id"] for q in results["processed_questions"]}
    
    # Process each question
    for question in questions_data["questions"]:
        if question["id"] in processed_ids:
            continue
            
        print(colored(f"\nProcessing question {question['id']}...", "cyan"))
        
        # Format question
        prompt = format_question(question)
        if not prompt:
            continue
        
        # Get model answer with retries
        model_answer = get_model_answer(client, prompt)
        if not model_answer:
            print(colored(f"Failed to get valid answer for question {question['id']} after {MAX_RETRIES} attempts", "red"))
            continue
        
        # Update results
        correct = model_answer == question["correct_answer"]
        results["processed_questions"].append({
            "id": question["id"],
            "model_answer": model_answer,
            "correct_answer": question["correct_answer"],
            "is_correct": correct,
            "timestamp": datetime.now().isoformat()
        })
        
        results["metadata"]["questions_processed"] += 1
        if correct:
            results["metadata"]["correct_answers"] += 1
        
        results["metadata"]["accuracy"] = (results["metadata"]["correct_answers"] / 
                                         results["metadata"]["questions_processed"]) * 100
        
        # Save progress
        save_results(results)
        
        # Print progress
        print_progress(results)
        
        # Small delay to avoid rate limits
        time.sleep(1)
    
    print(colored("\nTest run completed!", "green"))
    print_progress(results)

if __name__ == "__main__":
    main() 