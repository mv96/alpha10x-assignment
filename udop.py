from transformers import AutoProcessor, UdopForConditionalGeneration
from datasets import load_dataset
from PIL import Image
import torch
import evaluate
from bert_score import score
import time
from natsort import natsorted
import os


class DocumentEvaluator:
    def __init__(
        self,
        model_id="microsoft/udop-large",
        cache_directory="text_model_hf",
    ):
        self.cache_directory = cache_directory
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir=self.cache_directory,
            apply_ocr=True,
        )
        self.model = UdopForConditionalGeneration.from_pretrained(
            model_id, cache_dir=self.cache_directory
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # Initialize evaluation metrics
        self.rouge_metric = evaluate.load("rouge")
        self.bert_model_type = "bert-base-uncased"

        self.print_model_parameters()

    def print_model_parameters(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total number of parameters in the model: {total_params}")

    def evaluate_images(self, image_paths, question, ground_truth):
        best_answer = "unsure"
        best_confidence = 0
        best_page = -1
        results = []

        total_inference_start_time = time.time()

        for page_number, image_path in enumerate(image_paths, start=1):
            # Load and process image
            image = Image.open(image_path)

            # Prepare inputs
            encoding = self.processor(image, question, return_tensors="pt")

            # Move inputs to device
            encoding = {k: v.to(self.device) for k, v in encoding.items()}

            # Generate outputs
            outputs = self.model.generate(
                **encoding,
                output_scores=True,
                return_dict_in_generate=True,
                max_length=128,
            )

            # Calculate confidence scores
            probs = torch.stack(outputs.scores, dim=1).softmax(-1)
            gen_sequences = outputs.sequences[:, 1:]  # Skip the first token (BOS)
            gen_probs = torch.gather(probs, 2, gen_sequences[:, :, None]).squeeze(-1)
            sequence_confidence = gen_probs.prod(-1).item()

            # Decode the answer
            sequence = self.processor.batch_decode(
                outputs.sequences, skip_special_tokens=True
            )[0]

            # Store results
            results.append((page_number, sequence, sequence_confidence))
            print(
                f"Page {page_number}: Answer: {sequence}, Confidence: {sequence_confidence:.4f}"
            )

            # Update best answer if current score is higher
            if sequence_confidence > best_confidence:
                best_answer = sequence
                best_confidence = sequence_confidence
                best_page = page_number

        total_inference_time = time.time() - total_inference_start_time

        # Calculate metrics for best answer
        try:
            rouge_results = self.rouge_metric.compute(
                predictions=[best_answer],
                references=[ground_truth],
                use_stemmer=True,
            )
            rouge_l_score = rouge_results["rougeL"]
        except Exception as e:
            print(f"Error during ROUGE computation: {e}")
            rouge_l_score = 0.0

        try:
            P, R, F1 = score(
                [best_answer],
                [ground_truth],
                model_type=self.bert_model_type,
                lang="en",
                verbose=False,
            )
            bertscore_f1 = F1.mean().item()
        except Exception as e:
            print(f"Error during BERTScore computation: {e}")
            bertscore_f1 = 0.0

        # Print final results
        if best_confidence > 0.0:
            print(f"\nBest answer found on page {best_page}: {best_answer}")
            print(f"Confidence: {best_confidence:.4f}")
            print(f"ROUGE-L: {rouge_l_score * 100:.2f}%")
            print(f"BERTScore F1: {bertscore_f1 * 100:.2f}%")
            print(f"Total Inference Time: {total_inference_time:.4f} seconds")
        else:
            print("\nFinal answer: unsure")
            print("ROUGE-L: 0.00%")
            print("BERTScore F1: 0.00%")


def main():
    # Specify the directory containing the images
    image_directory = "output_images"

    # Get sorted list of image paths
    image_paths = natsorted(
        [
            os.path.join(image_directory, f)
            for f in os.listdir(image_directory)
            if f.endswith((".bmp", ".jpg", ".png"))
        ]
    )

    # Get question and ground truth
    raw_question = input("Please enter your question: ")
    # Prepend "Question answering." to the user's question
    question = f"Question answering. {raw_question}"
    ground_truth = input("Please enter the ground truth answer: ")

    evaluator = DocumentEvaluator()
    evaluator.evaluate_images(image_paths, question, ground_truth)


if __name__ == "__main__":
    main()
