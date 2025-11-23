# evaluate.py

import pandas as pd
import random
from typing import List, Optional, Literal
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from sentence_transformers import SentenceTransformer, util  # for semantic similarity

from .service import RAGService

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


class EvalQuery:
    def __init__(self, query: str, short_answer: str, long_answer: str):
        self.query = query
        self.short_answer = short_answer
        self.long_answer = long_answer


class Evaluator:
    """
    Evaluator for the RAG system using semantic similarity.
    Supports evaluating short answers, long answers, or both.
    """
    def __init__(
        self,
        rag_service: RAGService,
        csv_path: str,
        num_samples: int = 10,
        random_sample: bool = False,
        answer_type: Literal["short", "long", "both"] = "short",
        similarity_threshold: float = 0.7  # threshold for semantic match
    ):
        self.rag_service = rag_service
        self.csv_path = csv_path
        self.num_samples = num_samples
        self.random_sample = random_sample
        self.answer_type = answer_type
        self.similarity_threshold = similarity_threshold
        self.eval_queries: List[EvalQuery] = []
        self._load_eval_queries()
        # load multilingual embedding model
        self.embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    def _load_eval_queries(self):
        df = pd.read_csv(self.csv_path)

        required_cols = ["question", "short_answers", "long_answers"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing column '{col}' in dataset!")

        if self.random_sample:
            df = df.sample(n=min(self.num_samples, len(df)), random_state=RANDOM_SEED)
        else:
            df = df.head(self.num_samples)

        self.eval_queries = [
            EvalQuery(
                row["question"],
                row["short_answers"] if isinstance(row["short_answers"], str) else "",
                row["long_answers"] if isinstance(row["long_answers"], str) else ""
            )
            for _, row in df.iterrows()
        ]

    def _evaluate_single_answer(self, generated: str, expected: str):
        """
        Compute semantic similarity, BLEU, and ROUGE-L.
        """
        if not expected:
            return 0, 0.0, 0.0

        # ---- Semantic similarity ----
        gen_embed = self.embed_model.encode(generated, convert_to_tensor=True)
        exp_embed = self.embed_model.encode(expected, convert_to_tensor=True)
        similarity = util.cos_sim(gen_embed, exp_embed).item()
        correct = int(similarity >= self.similarity_threshold)

        # BLEU
        bleu_score = sentence_bleu(
            [expected.split()],
            generated.split(),
            weights=(0.5, 0.5)
        )

        # ROUGE-L
        rouge = Rouge()
        try:
            rouge_l = rouge.get_scores(generated, expected)[0]['rouge-l']['f']
        except Exception:
            rouge_l = 0.0

        return correct, bleu_score, rouge_l, similarity

    def run_evaluation(self) -> dict:
        metrics = {
            "short": {"precision": 0, "bleu": 0, "rouge_l": 0, "semantic_sim": 0, "n": 0},
            "long": {"precision": 0, "bleu": 0, "rouge_l": 0, "semantic_sim": 0, "n": 0},
        }

        detailed_results = []

        for item in self.eval_queries:
            rag_output = self.rag_service.ask_question(item.query)
            gen = rag_output.get("response", "")

            result_entry = {
                "query": item.query,
                "generated_answer": gen,
                "expected_short": item.short_answer,
                "expected_long": item.long_answer,
            }

            # ---- SHORT ANSWER ----
            if self.answer_type in ["short", "both"]:
                correct, bleu, rouge_l, sim = self._evaluate_single_answer(gen, item.short_answer)
                metrics["short"]["precision"] += correct
                metrics["short"]["bleu"] += bleu
                metrics["short"]["rouge_l"] += rouge_l
                metrics["short"]["semantic_sim"] += sim
                metrics["short"]["n"] += 1

                result_entry.update({
                    "correct_short": correct,
                    "bleu_short": bleu,
                    "rouge_l_short": rouge_l,
                    "semantic_sim_short": sim
                })

            # ---- LONG ANSWER ----
            if self.answer_type in ["long", "both"]:
                correct, bleu, rouge_l, sim = self._evaluate_single_answer(gen, item.long_answer)
                metrics["long"]["precision"] += correct
                metrics["long"]["bleu"] += bleu
                metrics["long"]["rouge_l"] += rouge_l
                metrics["long"]["semantic_sim"] += sim
                metrics["long"]["n"] += 1

                result_entry.update({
                    "correct_long": correct,
                    "bleu_long": bleu,
                    "rouge_l_long": rouge_l,
                    "semantic_sim_long": sim
                })

            detailed_results.append(result_entry)

        # Compute averages
        def finalize(m):
            if m["n"] == 0:
                return m
            m["precision"] /= m["n"]
            m["bleu"] /= m["n"]
            m["rouge_l"] /= m["n"]
            m["semantic_sim"] /= m["n"]
            return m

        metrics["short"] = finalize(metrics["short"])
        metrics["long"] = finalize(metrics["long"])

        return {
            "answer_type": self.answer_type,
            "metrics": metrics,
            "detailed": detailed_results
        }


if __name__ == "__main__":
    # Example usage
    from .app import rag_service

    evaluator = Evaluator(
        rag_service,
        csv_path=r"D:\DIGITISED TASK\rag_system\data\processed\Natural-Questions-with-metadata.csv",
        num_samples=10,
        random_sample=True,
        answer_type="both"
    )
    eval_results = evaluator.run_evaluation()
    print(eval_results)
