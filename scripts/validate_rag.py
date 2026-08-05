import requests
import json
import time

API_URL = "http://localhost:8000/api/query"

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


def run_validation():
    print(f"Starting RAG validation against {API_URL}...\n")

    results = []

    for i, question in enumerate(QUESTIONS, 1):
        print(f"[{i}/{len(QUESTIONS)}] Question: {question}")

        payload = {"question": question, "session_id": None}

        try:
            start_time = time.time()
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            elapsed_time = time.time() - start_time

            data = response.json()
            answer = data.get("answer", "No answer found")

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
