from termcolor import colored
import google.generativeai as genai
import os
import json
import time
from datetime import datetime
import signal
import sys

# Constants
RESULTS_FILE = "model_accuracy_results_gemini_flash.json"
DETAILED_RESPONSES_FILE = "model_detailed_responses_gemini_flash.json"
MODEL_NAME = "gemini-2.0-flash-thinking-exp-01-21"
MAX_RETRIES = 1000
RETRY_DELAY = 15

def init_gemini_client():
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 65536,
            "response_mime_type": "text/plain",
        }
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            generation_config=generation_config,
        )
        return model
    except Exception as e:
        print(colored(f"Error initializing Gemini client: {str(e)}", "red"))
        print(colored(f"Error details: {repr(e)}", "red"))
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

def save_detailed_responses(question_id, response):
    try:
        if os.path.exists(DETAILED_RESPONSES_FILE):
            with open(DETAILED_RESPONSES_FILE, "r", encoding="utf-8") as f:
                responses = json.load(f)
        else:
            responses = {
                "metadata": {
                    "model": MODEL_NAME,
                    "created_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat()
                },
                "responses": {}
            }

        responses["responses"][str(question_id)] = {
            "timestamp": datetime.now().isoformat(),
            "analysis": response
        }

        responses["metadata"]["last_updated"] = datetime.now().isoformat()

        with open(DETAILED_RESPONSES_FILE, "w", encoding="utf-8") as f:
            json.dump(responses, f, indent=2)

        print(colored(f"Saved detailed response for question {question_id}", "green"))
    except Exception as e:
        print(colored(f"Error saving detailed response: {e}", "red"))

def format_question(question_data):
    try:
        prompt = f"""Please solve this multiple choice question. You must provide your final answer in the format 'ANSWER: X' where X is A, B, C, or D at the end of your response.

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
        return None, None

    try:
        print(colored(f"Attempt {retry_count + 1}/{MAX_RETRIES + 1}...", "cyan"))
        chat = client.start_chat(history=[])
        response = chat.send_message(prompt)
        answer_text = response.text

        extracted_answer = extract_answer(answer_text)
        if extracted_answer:
            return extracted_answer, answer_text
        else:
            print(colored("Invalid answer format, retrying...", "yellow"))
            time.sleep(RETRY_DELAY)
            return get_model_answer(client, prompt, retry_count + 1)

    except Exception as e:
        print(colored(f"API error: {str(e)}", "red"))
        print(colored(f"Error details: {repr(e)}", "red"))
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

    print(colored("Initializing test run with Gemini Flash...", "green"))

    client = init_gemini_client()
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

        answer, response = get_model_answer(client, prompt)
        if not answer or not response:
            continue

        print(colored(f"Model answer: {answer}", "cyan"))

        # Save detailed response
        save_detailed_responses(question["id"], response)

        # Update results
        correct = answer == question["correct_answer"]
        results["processed_questions"].append({
            "id": question["id"],
            "model_answer": answer,
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