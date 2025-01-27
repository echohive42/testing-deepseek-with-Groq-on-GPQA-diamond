# DeepSeek-R1 Accuracy Testing on GPQA Diamond Dataset

This repository contains scripts for testing the DeepSeek-R1 model (70B Llama distilled version) accuracy on the GPQA Diamond dataset using the Groq API. The scripts evaluate the model's performance on a comprehensive set of graduate-level academic questions across various domains including physics, chemistry, and biology.

## Model Information

- **Model**: DeepSeek-R1 70B Llama Distilled
- **Provider**: Groq API
- **Description**: A distilled version of the DeepSeek-R1 70B model, optimized for efficient inference while maintaining strong performance on academic and reasoning tasks
- **Access**: Via Groq's API service

## Dataset Information

- **Name**: GPQA Diamond Dataset
- **Description**: A collection of 198 graduate-level multiple-choice questions designed to test advanced understanding across scientific domains
- **Domains**: Physics, Chemistry, Biology, and other scientific fields
- **Difficulty**: Graduate to post-graduate level questions
- **Format**: Multiple choice (A, B, C, D options)

## ❤️ Join my AI community & Get 400+ AI Projects

This is one of 400+ fascinating projects in my collection! [Support me on Patreon](https://www.patreon.com/c/echohive42/membership) to get:

- 🎯 Access to 400+ AI projects (and growing daily!)
  - Including advanced projects like [2 Agent Real-time voice template with turn taking](https://www.patreon.com/posts/2-agent-real-you-118330397)
- 📥 Full source code & detailed explanations
- 📚 1000x Cursor Course
- 🎓 Live coding sessions & AMAs
- 💬 1-on-1 consultations (higher tiers)

## Prerequisites

```bash
pip install groq termcolor
```

Set your Groq API key as an environment variable:

```bash
export GROQ_API_KEY='your-api-key'
```

## Scripts Overview

### 1. test_model_accuracy.py

Basic script that tests model accuracy on multiple-choice questions.

#### Features:

- Processes questions from `gpqa_questions.json`
- Saves results to `model_accuracy_results.json`
- Implements retry logic for API errors
- Real-time progress tracking
- Resume capability for interrupted runs

#### Usage:

```bash
python test_model_accuracy.py
```

#### Output Format:

```json
{
  "metadata": {
    "model": "deepseek-r1-distill-llama-70b",
    "start_time": "...",
    "last_updated": "...",
    "total_questions": 198,
    "questions_processed": 50,
    "correct_answers": 35,
    "accuracy": 70.0
  },
  "processed_questions": [
    {
      "id": 1,
      "model_answer": "A",
      "correct_answer": "A",
      "is_correct": true,
      "timestamp": "..."
    }
  ]
}
```

### 2. test_model_accuracy_with_verification.py

Enhanced version that includes a verification step for each answer.

#### Additional Features:

- Two-step process: initial answer + verification
- Tracks both original and verified answers
- Counts answers changed by verifier
- More detailed results tracking
- Same retry and resume capabilities as basic version

#### Usage:

```bash
python test_model_accuracy_with_verification.py
```

#### Output Format:

```json
{
  "metadata": {
    "model": "deepseek-r1-distill-llama-70b",
    "start_time": "...",
    "last_updated": "...",
    "total_questions": 198,
    "questions_processed": 50,
    "correct_answers": 35,
    "accuracy": 70.0,
    "answers_changed_by_verifier": 5
  },
  "processed_questions": [
    {
      "id": 1,
      "original_answer": "A",
      "verified_answer": "B",
      "was_changed": true,
      "correct_answer": "B",
      "is_correct": true,
      "timestamp": "..."
    }
  ]
}
```

## Configuration

Both scripts use the following constants that can be modified:

```python
RESULTS_FILE = "model_accuracy_results.json"  # or model_accuracy_results_verified.json
MODEL_NAME = "deepseek-r1-distill-llama-70b"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
```

## Error Handling

- Retries API calls up to 3 times
- Handles invalid answer formats
- Graceful shutdown with Ctrl+C
- Preserves progress on interruption
- Detailed error messages with color coding

## Progress Tracking

Both scripts provide real-time progress updates:

- Questions processed
- Correct answers
- Current accuracy
- Verification changes (in verified version)

## Files

- `gpqa_questions.json`: Input questions dataset
- `model_accuracy_results.json`: Results from basic script
- `model_accuracy_results_verified.json`: Results from verified script

## Notes

- The verification script takes approximately twice as long to run due to the additional API call for verification
- Both scripts can be safely interrupted and resumed
- Results are saved after each question
- Use the verified version for higher accuracy but slower processing
- Use the basic version for quicker results without verification

## Error Messages

Common error messages and their meanings:

- "Invalid answer format": Model response doesn't contain expected format
- "API error": Connection or rate limit issues
- "Maximum retry attempts reached": Failed after 3 retries
