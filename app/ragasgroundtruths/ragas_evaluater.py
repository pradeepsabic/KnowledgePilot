from datasets import Dataset
import asyncio
from dotenv import load_dotenv
import os
import sys
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from app.crews.knowledgepilot_crew import KnowledgePilotCrew

def executerag_crew(query: str):
    try:
        crew = KnowledgePilotCrew(query)
        result = crew.kickoff()
        answer = str(result)
        contexts = [answer]
        return {"answer": answer, "contexts": contexts}
    except Exception as e:
        print(f"Error processing query '{query}': {e}")
        return {"answer": "Error", "contexts": []}

async def main():
    GROUNDTRUTH_PATH = "app/ragasgroundtruths/groundtruths.jsonl"
    if not os.path.exists(GROUNDTRUTH_PATH):
        print(f"Error: Dataset not found at {GROUNDTRUTH_PATH}")
        return

    dataset = Dataset.from_json(GROUNDTRUTH_PATH)

    results = []
    questions = []
    ground_truths = []

    for entry in dataset:
        question = entry["question"]
        ground_truth = entry["ground_truth"]
        print(f"Processing: {question[:80]}...")
        
        output = executerag_crew(question)
        results.append(output)
        questions.append(question)
        ground_truths.append(ground_truth)

    eval_data = {
        "question": questions,
        "answer": [r["answer"] for r in results],
        "contexts": [r["contexts"] for r in results],
        "ground_truth": ground_truths,
    }

    eval_dataset = Dataset.from_dict(eval_data)

    metrics = [
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    ]

    result = evaluate(dataset=eval_dataset, metrics=metrics)

    print("Evaluation Results:")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())