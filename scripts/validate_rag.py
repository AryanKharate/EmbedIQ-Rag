"""
scripts/validate_rag.py

Manual smoke-test script: fires a fixed list of questions at a running
/api/query endpoint and saves the answers for a human to eyeball.

Not a substitute for scripts/squad_eval.py (which scores accuracy/recall
against a labeled dataset) — this is for a quick "does the app still answer
sensibly" check after a change, e.g.:

    docker compose exec web python scripts/validate_rag.py

Requires a running stack with at least one document ingested.
"""

import json
import time

import requests

BASE_URL = "http://localhost:8000"
API_URL = f"{BASE_URL}/api/query"

# A disposable local account used only to obtain a JWT for this script.
# Registration is skipped (409) on every run after the first.
VALIDATION_EMAIL = "validate-rag@local.test"
VALIDATION_PASSWORD = "validate-rag-smoke-test-only"

QUESTIONS = [
    "What is the role of DNS in internet communication?",
    "Why are GPUs important for artificial intelligence and deep learning?",
    "What is cloud computing, and what are its advantages?",
    "Compare Hard Disk Drives (HDDs) and Solid-State Drives (SSDs).",
    "How does the Input → Process → Store → Output cycle work?",
    "What is the difference between relational and NoSQL databases?",
    "What is an operating system, and what are its main responsibilities?",
    "Why was the invention of the transistor a turning point in computer history?",
    "What is load balancing, and why is it important?",
    "Explain the differences between IaaS, PaaS, and SaaS.",
    "What are the four basic functions performed by a computer?",
    "What is Kubernetes, and why is it used?",
    "What is the purpose of an IP address?",
    "How do operating systems manage multiple applications running simultaneously?",
    "Why are SSDs considered better than HDDs for modern computers?",
    "What is the role of the motherboard in a computer system?",
    "Explain the evolution of computers from the abacus to modern AI systems.",
    "What are containers, and how are they different from virtual machines?",
    "What is SQL used for?",
    "Compare Windows, macOS, and Linux.",
]


def get_access_token() -> str:
    """
    Log in with a disposable local account, registering it first if it
    doesn't exist yet. /api/query requires a JWT bearer token.
    """
    login = requests.post(
        f"{BASE_URL}/api/auth/login",
        json={"email": VALIDATION_EMAIL, "password": VALIDATION_PASSWORD},
    )
    if login.ok:
        return login.json()["access"]

    register = requests.post(
        f"{BASE_URL}/api/auth/register",
        json={
            "email": VALIDATION_EMAIL,
            "password": VALIDATION_PASSWORD,
            "display_name": "RAG Validation",
        },
    )
    register.raise_for_status()
    return register.json()["access"]


def ask(question: str, token: str) -> str:
    """
    POST a question and consume the SSE stream, returning the concatenated
    answer text. /api/query streams three event types:
      sources -> token* -> done  (see apps/generation/api.py)
    """
    response = requests.post(
        API_URL,
        json={"question": question, "session_id": None},
        headers={"Authorization": f"Bearer {token}"},
        stream=True,
    )
    response.raise_for_status()

    answer_parts: list[str] = []
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        event = json.loads(line[len("data: ") :])
        if event["type"] == "token":
            answer_parts.append(event["text"])
        elif event["type"] == "done":
            break

    return "".join(answer_parts)


def run_validation():
    print(f"Starting RAG validation against {API_URL}...\n")
    token = get_access_token()

    results = []

    for i, question in enumerate(QUESTIONS, 1):
        print(f"[{i}/{len(QUESTIONS)}] Question: {question}")

        try:
            start_time = time.time()
            answer = ask(question, token)
            elapsed_time = time.time() - start_time

            print(f"Answer ({elapsed_time:.2f}s): {answer}\n")
            print("-" * 80 + "\n")

            results.append(
                {"question": question, "answer": answer, "time_seconds": elapsed_time}
            )

        except requests.exceptions.RequestException as e:
            print(f"Error querying API: {e}\n")
            print("-" * 80 + "\n")
            results.append({"question": question, "error": str(e)})

    # Save results to a file for review
    output_file = "rag_validation_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Validation complete! Detailed results saved to {output_file}")


if __name__ == "__main__":
    run_validation()
