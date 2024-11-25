from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import torch
import os
from natsort import natsorted
import evaluate
from bert_score import score  # Importing BERTScore
import time


class ModelEvaluator:
    def __init__(
        self,
        model_id,
        ocr_output_dir,
        custom_cache_dir="./text_models_hf",
    ):
        self.model_id = model_id
        self.ocr_output_dir = ocr_output_dir
        self.custom_cache_dir = custom_cache_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_index = 0 if torch.cuda.is_available() else -1

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, cache_dir=custom_cache_dir
        )
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            model_id, cache_dir=custom_cache_dir
        ).to(self.device)
        self.qa_pipeline = pipeline(
            "question-answering",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device_index,
        )

        # Initialize ROUGE evaluation metric
        self.rouge_metric = evaluate.load("rouge")

        # Initialize BERTScore parameters
        self.bert_model_type = "bert-base-uncased"  # You can choose other models

    def evaluate(self, text_files, question, ground_truth):
        best_answer = "unsure"
        best_confidence = 0
        best_page = -1

        # Measure total inference time
        total_inference_start_time = time.time()  # Start timing

        for page_number, text_filename in enumerate(text_files, start=1):
            file_path = os.path.join(self.ocr_output_dir, text_filename)
            with open(file_path, "r", encoding="utf-8") as file:
                context = file.read()

            # Measure inference time for the entire loop
            try:
                result = self.qa_pipeline(question=question, context=context)
            except ValueError as ve:
                print(f"Page {page_number}: Error - {ve}")
                continue

            # Check if the result is empty
            if not result:
                print(f"Page {page_number}: No answer found.")
                continue  # Skip to the next page

            # Extract answer and score
            if isinstance(result, list) and len(result) > 0:
                answer = result[0].get("answer", "No answer found.")
                score_confidence = result[0].get("score", 0.0)
            elif isinstance(result, dict):
                answer = result.get("answer", "No answer found.")
                score_confidence = result.get("score", 0.0)
            else:
                print(f"Page {page_number}: Unexpected result format: {result}")
                continue

            # Commenting out the intermediate output
            # print(f"Page {page_number}: Answer: {answer}, Score: {score_confidence:.4f}")

            # Update the best answer if the current score is higher
            if score_confidence > best_confidence:
                best_answer = answer
                best_confidence = score_confidence
                best_page = page_number

        total_inference_end_time = time.time()  # End timing
        total_inference_time = (
            total_inference_end_time - total_inference_start_time
        )  # Calculate total inference time

        # Use original best answer and ground truth for evaluation
        normalized_best_answer = best_answer  # No normalization
        normalized_ground_truth = ground_truth  # No normalization

        # Compute ROUGE-L score only
        try:
            rouge_results = self.rouge_metric.compute(
                predictions=[normalized_best_answer],
                references=[normalized_ground_truth],
                use_stemmer=True,
            )
            # Access ROUGE-L score directly
            rouge_l_score = rouge_results["rougeL"]  # This should be a float
        except Exception as e:
            print(f"Error during ROUGE computation: {e}")
            rouge_l_score = 0.0

        # Compute BERTScore
        try:
            P, R, F1 = score(
                [normalized_best_answer],
                [normalized_ground_truth],
                model_type=self.bert_model_type,
                lang="en",
                verbose=False,
            )
            bertscore_f1 = F1.mean().item()
        except Exception as e:
            print(f"Error during BERTScore computation: {e}")
            bertscore_f1 = 0.0

        # Final output based on the metrics
        if best_confidence > 0.0:
            print(
                f"\nBest answer found on page {best_page}: {best_answer} with confidence {best_confidence:.2f}"
            )
            print(f"ROUGE-L: {rouge_l_score * 100:.2f}%")  # Report only ROUGE-L
            print(f"BERTScore F1: {bertscore_f1 * 100:.2f}%")
            print(
                f"Total Inference Time: {total_inference_time:.4f} seconds"
            )  # Print total inference time
            print(
                f"Model Weight Count: {self.model.num_parameters() / 1_000_000:.2f} million"
            )  # Print model weight count in millions
        else:
            print("\nFinal answer: unsure")
            print("ROUGE-L: 0.00%")
            print("BERTScore F1: 0.00%")


def main():
    # List of model identifiers to evaluate
    model_ids = [
        "twmkn9/distilbert-base-uncased-squad2",
        "deepset/bert-large-uncased-whole-word-masking-squad2",
        "deepset/roberta-base-squad2",
        "allenai/longformer-base-4096",
    ]

    # Specify the directory containing the OCR output text files
    ocr_output_dir = "./ocr_output"

    # Define the question and ground truth dynamically

    # What is the expected size of battery recycling in 2030?
    question = input("Please enter your question: ")

    # The global EV battery recycling market is projected to exceed $46.5 billion by 2030, driven by the rapid growth in electric vehicles and an increasing wave of battery retirements​
    ground_truth = input("Please enter the ground truth answer: ")

    # Get a sorted list of all text files in the ocr_output directory using natsorted
    text_files = natsorted(
        [f for f in os.listdir(ocr_output_dir) if f.endswith(".txt")]
    )

    # Iterate through each model and evaluate
    for model_id in model_ids:
        print(f"Evaluating model: {model_id}")
        evaluator = ModelEvaluator(model_id, ocr_output_dir)
        evaluator.evaluate(text_files, question, ground_truth)


if __name__ == "__main__":
    main()
