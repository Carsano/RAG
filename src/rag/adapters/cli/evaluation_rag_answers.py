"""
CLI for model evaluation
"""
# src/rag/adapters/cli/evaluate_rag_answers.py
from src.rag.application.use_cases.rag_evaluation import RagEvaluationUseCase


def main():
    uc = RagEvaluationUseCase(
        input_path="logs/interactions/interactions.jsonl",
        out_root="logs/evaluations/ragas_eval",
        llm_max_retries=3,
        retry_passes=2,
        retry_sleep_seconds=60,
    )
    uc.run()


if __name__ == "__main__":
    main()
