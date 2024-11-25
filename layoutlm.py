import warnings

warnings.filterwarnings("ignore")

from transformers import LayoutLMv3Processor, LayoutLMv3ForQuestionAnswering
import torch
import os
from natsort import natsorted
import evaluate
from bert_score import score  # Importing BERTScore
import pytesseract
from PIL import Image
import torch.nn.functional as F
import time
from tqdm import tqdm  # Added tqdm for progress bar


# "rubentito/layoutlmv3-base-mpdocvqa"
class LayoutLMv3Evaluator:
    def __init__(
        self,
        model_id="rubentito/layoutlmv3-base-mpdocvqa",
        ocr_output_dir="./ocr_output",
        image_dir="./output_images",
        custom_cache_dir="./text_models_hf",
    ):
        self.model_id = model_id
        self.ocr_output_dir = ocr_output_dir
        self.image_dir = image_dir
        self.custom_cache_dir = custom_cache_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_index = 0 if torch.cuda.is_available() else -1

        # Load LayoutLMv3 processor and model
        self.processor = LayoutLMv3Processor.from_pretrained(
            self.model_id, apply_ocr=False, cache_dir=self.custom_cache_dir
        )
        self.model = LayoutLMv3ForQuestionAnswering.from_pretrained(
            self.model_id, cache_dir=self.custom_cache_dir
        ).to(self.device)

        # Initialize ROUGE evaluation metric
        self.rouge_metric = evaluate.load("rouge")

        # Initialize BERTScore parameters
        self.bert_model_type = "bert-base-uncased"  # You can choose other models

    def perform_ocr(self, image_path):
        image = Image.open(image_path).convert("RGB")
        original_width, original_height = image.size  # Get original image dimensions

        # Calculate scale factors to fit the width and height within 1000
        width_scale = 1000 / original_width
        height_scale = 1000 / original_height

        # Perform OCR to get words and their bounding boxes
        ocr_data = pytesseract.image_to_data(
            image, output_type=pytesseract.Output.DICT, lang="eng"
        )

        words = []
        boxes = []

        n_boxes = len(ocr_data["level"])
        for i in range(n_boxes):
            word = ocr_data["text"][i].strip()
            if word:  # Ignore empty strings
                words.append(word)
                (x, y, w, h) = (
                    ocr_data["left"][i],
                    ocr_data["top"][i],
                    ocr_data["width"][i],
                    ocr_data["height"][i],
                )
                # Scale bounding boxes
                x0 = int(x * width_scale)
                y0 = int(y * height_scale)
                x1 = int((x + w) * width_scale)
                y1 = int((y + h) * height_scale)

                # Ensure that the coordinates do not exceed 1000
                x0 = min(max(x0, 0), 1000)
                y0 = min(max(y0, 0), 1000)
                x1 = min(max(x1, 0), 1000)
                y1 = min(max(y1, 0), 1000)

                boxes.append([x0, y0, x1, y1])

        # Verify that the number of words matches the number of bounding boxes
        if len(words) != len(boxes):
            raise ValueError(
                f"The number of words ({len(words)}) does not match the number of bounding boxes ({len(boxes)})."
            )

        return image, words, boxes

    def evaluate(self, image_files, question, ground_truth):
        best_answer = "unsure"
        best_confidence = 0
        best_page = -1

        # Measure total inference time
        total_inference_start_time = time.time()  # Start timing

        # Initialize progress bar
        for page_number, image_filename in enumerate(
            tqdm(image_files, desc="Processing pages"), start=1
        ):
            image_path = os.path.join(self.image_dir, image_filename)
            try:
                image, words, boxes = self.perform_ocr(image_path)
            except Exception as e:
                print(f"Page {page_number}: OCR Error - {e}")
                continue

            document_encoding = self.processor(
                image,
                question,
                words,
                boxes=boxes,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(self.device)

            try:
                outputs = self.model(**document_encoding)
            except Exception as e:
                print(f"Page {page_number}: Model Inference Error - {e}")
                continue

            # Apply Softmax to get probabilities
            start_probs = F.softmax(outputs.start_logits, dim=1)
            end_probs = F.softmax(outputs.end_logits, dim=1)

            # Get the most probable start and end indices
            start_idx = torch.argmax(start_probs, dim=1)
            end_idx = torch.argmax(end_probs, dim=1)

            # Extract the probabilities of the selected start and end indices
            start_confidence = start_probs[0, start_idx].item()
            end_confidence = end_probs[0, end_idx].item()

            # Calculate overall confidence as the average of start and end confidences
            overall_confidence = (start_confidence + end_confidence) / 2

            # Decode the answer
            answers = self.processor.tokenizer.decode(
                document_encoding["input_ids"][0][start_idx : end_idx + 1]
            ).strip()

            # Print the answer and confidence for the current page
            print(
                f'Page {page_number}: Answer: "{answers}" with Confidence: {overall_confidence * 100:.2f}%'
            )

            # Update the best answer if the current score is higher
            if overall_confidence > best_confidence:
                best_answer = answers
                best_confidence = overall_confidence
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
                f'\nBest answer found on page {best_page}: "{best_answer}" with confidence {best_confidence * 100:.2f}%'
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
    # Specify the directory containing the OCR output text files
    image_dir = "./output_images"

    # Define the question and ground truth dynamically

    # Example Question: What is the expected size of battery recycling in 2030?
    question = input("Please enter your question: ")

    # Example Ground Truth: The global EV battery recycling market is projected to exceed $46.5 billion by 2030, driven by the rapid growth in electric vehicles and an increasing wave of battery retirements.
    ground_truth = input("Please enter the ground truth answer: ")

    # Get a sorted list of all image files in the image_dir using natsorted
    image_files = natsorted(
        [
            f
            for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ]
    )

    evaluator = LayoutLMv3Evaluator()
    evaluator.evaluate(image_files, question, ground_truth)


if __name__ == "__main__":
    main()
