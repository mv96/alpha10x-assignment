import re
import torch
import os
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
import evaluate
from bert_score import score  # Importing BERTScore
import time
from natsort import natsorted  # Importing natsorted for natural sorting


class DocumentEvaluator:
    def __init__(self, model_id, cache_directory="./text_model_hf"):
        self.cache_directory = cache_directory
        self.processor = DonutProcessor.from_pretrained(
            model_id, cache_dir=self.cache_directory
        )
        self.model = VisionEncoderDecoderModel.from_pretrained(
            model_id, cache_dir=self.cache_directory
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # Initialize ROUGE evaluation metric
        self.rouge_metric = evaluate.load("rouge")

        # Initialize BERTScore parameters
        self.bert_model_type = "bert-base-uncased"  # You can choose other models

    def evaluate_images(self, image_paths, question, ground_truth):
        best_answer = "unsure"
        best_confidence = 0
        best_page = -1
        results = []

        # Measure total inference time
        total_inference_start_time = time.time()  # Start timing

        for page_number, image_path in enumerate(image_paths, start=1):
            # Load test image from the specified path
            image = Image.open(image_path)

            # Prepare decoder inputs
            task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"
            prompt = task_prompt.replace("{user_input}", question)
            decoder_input_ids = self.processor.tokenizer(
                prompt, add_special_tokens=False, return_tensors="pt"
            ).input_ids

            pixel_values = self.processor(image, return_tensors="pt").pixel_values

            # Generate outputs
            outputs = self.model.generate(
                pixel_values.to(self.device),
                decoder_input_ids=decoder_input_ids.to(self.device),
                max_length=self.model.decoder.config.max_position_embeddings,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                output_scores=True,
                return_dict_in_generate=True,
            )

            # Decode the generated sequence
            sequence = self.processor.batch_decode(outputs.sequences)[0]
            sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(
                self.processor.tokenizer.pad_token, ""
            )

            # Assuming decoder_output.scores is a list of logits for each generated token
            probs = torch.stack(outputs.scores, dim=1).softmax(-1)
            gen_sequences = outputs.sequences[:, decoder_input_ids.shape[-1] : -1]
            gen_probs = torch.gather(probs, 2, gen_sequences[:, :, None]).squeeze(-1)
            sequence_confidence = gen_probs.prod(-1).item()

            # Store results for evaluation
            results.append((page_number, sequence, sequence_confidence))
            print(
                f"Page {page_number}: Answer: {sequence}, Confidence: {sequence_confidence:.4f}"
            )

            # Update the best answer if the current score is higher
            if sequence_confidence > best_confidence:
                best_answer = sequence
                best_confidence = sequence_confidence
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
        else:
            print("\nFinal answer: unsure")
            print("ROUGE-L: 0.00%")
            print("BERTScore F1: 0.00%")


def main():
    # Specify the model identifier
    model_id = "naver-clova-ix/donut-base-finetuned-docvqa"

    # Specify the directory containing the images
    image_directory = "output_images"

    # Get a sorted list of all image paths in the directory using natsorted
    image_paths = natsorted(
        [
            os.path.join(image_directory, f)
            for f in os.listdir(image_directory)
            if f.endswith((".bmp", ".jpg", ".png"))
        ]
    )

    # Define the question and ground truth dynamically
    question = input("Please enter your question: ")
    ground_truth = input("Please enter the ground truth answer: ")

    evaluator = DocumentEvaluator(model_id)
    evaluator.evaluate_images(image_paths, question, ground_truth)


if __name__ == "__main__":
    main()
