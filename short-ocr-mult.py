import evaluate
import json
import os
import numpy as np
import requests
import time
from datetime import datetime
import base64
from pathlib import Path
from PIL import Image
import io
from natsort import natsorted


class ContextualLLMEvaluator:
    def __init__(self):
        self.rouge = evaluate.load("rouge")
        self.bert_score = evaluate.load("bertscore")
        self.ollama_api = "http://localhost:11434/api/generate"

        # Removed global context loading
        print("Context will be loaded per image.")

    def encode_image_to_base64(self, image_path):
        """Convert image to base64 string"""
        try:
            with Image.open(image_path) as img:
                # Convert image to RGB if it's not
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Convert to JPEG format in memory
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG")
                img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
                return img_str
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None

    def load_test_data(self, json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
            print("test_data: ", data)
            return data

    def get_model_response(self, model_name, prompt):
        """Get response from Ollama API with stream handling and timing"""
        data = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.1,
        }

        try:
            # Start timing
            start_time = time.time()

            # Make API call
            response = requests.post(self.ollama_api, json=data)
            response.raise_for_status()

            # End timing
            end_time = time.time()
            inference_time = end_time - start_time

            # Parse the response
            response_json = response.json()
            if "response" in response_json:
                return response_json["response"].strip(), inference_time
            else:
                print(f"Unexpected response format: {response_json}")
                return None, inference_time

        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            return None, 0
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Response content: {response.text[:200]}...")
            return None, 0
        except Exception as e:
            print(f"Error getting model response: {e}")
            return None, 0

    def get_model_response_with_image(self, model_name, prompt, image_path):
        """Get response from Ollama API for image input with relevance and confidence scoring"""
        img_base64 = self.encode_image_to_base64(image_path)
        if not img_base64:
            return None, 0, 0, 0

        # Set improved prompt as per user instructions
        improved_prompt = f"""
{prompt}\n
Analyze the provided text and image to evaluate their relevance and confidence for answering the question.
Your response must include:
1. A relevance score (0 to 1): Assess how closely the text and image address the question, being strict in awarding high scores. Only assign a high score if the alignment is exceptionally clear and strong. Assign a low score if the image and text do not provide specific information about the question asked.
2. A confidence score (0 to 1): Indicate your confidence in the accuracy of the answer, being strict in awarding high scores. Only assign a high score if the answer is fully supported by the data provided.
3. The final answer to the question (without any commentary on the scores).

Use the following format for your response:
Relevance score: [score]
Confidence score: [score]
Answer: [your answer]
"""
        data = {
            "model": model_name,
            "prompt": improved_prompt,
            "stream": False,
            "temperature": 0,
            "images": [img_base64],
        }

        try:
            start_time = time.time()
            response = requests.post(self.ollama_api, json=data)
            response.raise_for_status()
            end_time = time.time()
            inference_time = end_time - start_time

            response_json = response.json()
            if "response" in response_json:
                response_text = response_json["response"].strip()

                # Remove any triple backticks if present
                response_text = response_text.replace("```", "").strip()

                # Extract relevance score, confidence score, and answer
                try:
                    lines = response_text.split("\n")
                    # Ensure there are at least three lines
                    if len(lines) < 3:
                        raise ValueError("Incomplete response from model.")

                    relevance_line = lines[0]
                    confidence_line = lines[1]
                    answer_lines = lines[2:]

                    relevance_score = float(
                        relevance_line.replace("Relevance score:", "").strip()
                    )
                    confidence_score = float(
                        confidence_line.replace("Confidence score:", "").strip()
                    )
                    # Combine all answer lines into a single answer
                    answer = " ".join(
                        [
                            (
                                line.replace("Answer:", "").strip()
                                if "Answer:" in line
                                else line.strip()
                            )
                            for line in answer_lines
                        ]
                    )

                    return answer, inference_time, relevance_score, confidence_score
                except Exception as parse_e:
                    print(
                        f"Error parsing scores from response: {response_text} | {parse_e}"
                    )
                    return response_text, inference_time, 0, 0
            else:
                print(f"Unexpected response format: {response_json}")
                return None, inference_time, 0, 0

        except Exception as e:
            print(f"Error getting model response for image: {e}")
            return None, 0, 0, 0

    def get_model_predictions(self, model_name, test_data):
        predictions = []
        references = []
        inference_times = []
        relevance_scores = []
        confidence_scores = []

        # Get list of images from output_images directory and sort them in natural order
        image_dir = Path("output_images")
        image_files = natsorted(image_dir.glob("*.bmp"))

        for image_path in image_files:
            try:
                # Derive corresponding OCR text file path
                ocr_filename = image_path.stem + ".txt"
                ocr_path = Path("ocr_output") / ocr_filename

                # Load OCR context
                if ocr_path.exists():
                    with open(ocr_path, "r", encoding="utf-8") as f:
                        context = f.read().strip()
                    print(f"Loaded context from {ocr_path}")
                else:
                    print(f"OCR file {ocr_path} does not exist. Using empty context.")
                    context = ""

                # Create prompt with context and question for each test data item
                for item in test_data:
                    try:
                        print(
                            f"\nProcessing question for {model_name}: {item['input']}"
                        )

                        # Create prompt with context and question
                        prompt = f"""OCR scanned text: {context} 
Question: {item['input']}"""

                        # Get prediction using vision model
                        prediction, inference_time, relevance, confidence = (
                            self.get_model_response_with_image(
                                model_name, prompt, image_path
                            )
                        )

                        if prediction:
                            predictions.append(prediction)
                            references.append(item["reference"])
                            inference_times.append(inference_time)
                            relevance_scores.append(relevance)
                            confidence_scores.append(confidence)

                            print(f"\nImage: {image_path}")
                            print(f"Prediction: {prediction}")
                            print(f"Relevance Score: {relevance}")
                            print(f"Confidence Score: {confidence}")
                            print(f"Inference Time: {inference_time:.4f} seconds")
                        else:
                            print(
                                f"No valid response received from model for image {image_path}"
                            )

                    except Exception as e:
                        print(f"Error processing question: {e}")
                        import traceback

                        print(traceback.format_exc())
                        continue

            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                import traceback

                print(traceback.format_exc())
                continue

        return (
            predictions,
            references,
            inference_times,
            relevance_scores,
            confidence_scores,
        )

    def compute_metrics(self, predictions, references):
        if not predictions or not references:
            print("No predictions or references to evaluate")
            return None

        try:
            # BERT scores
            bert_scores = self.bert_score.compute(
                predictions=predictions,
                references=references,
                model_type="bert-base-uncased",
            )

            # Average BERT F1 score
            avg_bert_f1 = float(np.mean(bert_scores["f1"]))

            return {"bert_f1": avg_bert_f1}

        except Exception as e:
            print(f"Error computing metrics: {e}")
            return None

    def get_model_metadata(self, model_name):
        """Retrieve model metadata from Ollama API."""
        metadata_api = "http://localhost:11434/api/model_metadata"  # Example endpoint
        try:
            response = requests.get(f"{metadata_api}/{model_name}")
            response.raise_for_status()
            metadata = response.json()
            print(f"Metadata for {model_name}: {metadata}")  # Debugging

            # Check for possible keys
            if "parameters" in metadata:
                param_count = metadata["parameters"]
            elif "parameter_count" in metadata:
                param_count = metadata["parameter_count"]
            else:
                param_count = "N/A"
                print(f"Parameter count key not found in metadata for {model_name}.")

            return param_count
        except requests.exceptions.RequestException as e:
            print(f"Error fetching metadata for {model_name}: {e}")
            return "Unknown"
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response for {model_name}: {e}")
            return "Unknown"

    def evaluate_model(self, model_name, test_data):
        print(f"\nEvaluating {model_name}...")

        # Fetch parameter count dynamically
        param_count = self.get_model_metadata(model_name)
        print(f"Model Name: {model_name}")
        if param_count not in ["Unknown", "N/A"]:
            print(f"Parameter Count: {param_count} Billion Parameters")
        else:
            print("Parameter Count: Not Available")

        total_inference_time = 0  # Initialize total inference time

        try:
            # Get predictions with timing
            (
                predictions,
                references,
                inference_times,
                relevance_scores,
                confidence_scores,
            ) = self.get_model_predictions(model_name, test_data)

            if not predictions or not references:
                print(f"No predictions generated for {model_name}")
                return None

            # Combine all metrics for sorting
            combined_results = list(
                zip(
                    predictions,
                    references,
                    inference_times,
                    relevance_scores,
                    confidence_scores,
                )
            )

            # Sort based on Relevance Score (descending) and then Confidence Score (descending)
            combined_results.sort(key=lambda x: (x[3], x[4]), reverse=True)

            # Unzip the sorted results
            (
                sorted_predictions,
                sorted_references,
                sorted_inference_times,
                sorted_relevance_scores,
                sorted_confidence_scores,
            ) = zip(*combined_results)

            # Compute BERT F1 scores
            bert_metrics = self.compute_metrics(sorted_predictions, sorted_references)

            # Compute ROUGE-L score on the best prediction only
            best_prediction = sorted_predictions[0]
            best_reference = sorted_references[0]
            rouge_scores_best = self.rouge.compute(
                predictions=[best_prediction], references=[best_reference]
            )
            rouge_l_best = rouge_scores_best.get("rougeL", {}).get("fmeasure", 0.0)

            if bert_metrics:
                # Calculate average inference time
                avg_inference_time = sum(sorted_inference_times) / len(
                    sorted_inference_times
                )

                # Calculate total inference time
                total_inference_time = sum(sorted_inference_times)

                # Save detailed results
                results_dir = "evaluation_results"
                os.makedirs(results_dir, exist_ok=True)

                # Get timestamp for the evaluation
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # Prepare sorted detailed results
                sorted_detailed_results = []
                for idx, (pred, ref, inf_time, rel, conf) in enumerate(
                    combined_results, 1
                ):
                    sorted_detailed_results.append(
                        {
                            "page_number": idx,  # Added page number
                            "prediction": pred,
                            "reference": ref,
                            "inference_time": inf_time,
                            "relevance_score": rel,
                            "confidence_score": conf,
                        }
                    )

                detailed_results = {
                    "model_name": model_name,
                    "parameter_count_billion": param_count,
                    "timestamp": timestamp,
                    "sorted_predictions": sorted_detailed_results,
                    "metrics": bert_metrics,
                    "rougeL_best_prediction": rouge_l_best,
                    "inference_times": {
                        "individual": sorted_inference_times,
                        "average": avg_inference_time,
                        "min": min(sorted_inference_times),
                        "max": max(sorted_inference_times),
                        "total": total_inference_time,  # Added total inference time
                    },
                }

                safe_model_name = model_name.replace(":", "_").replace("/", "_")
                result_file = (
                    f"{results_dir}/{safe_model_name}_{timestamp}_sorted_results.json"
                )
                with open(result_file, "w") as f:
                    json.dump(detailed_results, f, indent=4)

                print(f"\n{model_name} Sorted Results:")
                for result in sorted_detailed_results:
                    print(f"\nPage {result['page_number']}:")
                    print(f"Relevance Score: {result['relevance_score']}")
                    print(f"Confidence Score: {result['confidence_score']}")
                    print(f"Inference Time: {result['inference_time']:.4f} seconds")
                    print(f"Prediction: {result['prediction']}")

                print("\nPerformance Metrics:")
                print("Average BERT F1 Score:", bert_metrics["bert_f1"])
                print("ROUGE-L Score on Best Prediction:", rouge_l_best)
                print("\nInference Time Statistics:")
                print(f"Average: {avg_inference_time:.4f} seconds")
                print(
                    f"Total: {total_inference_time:.4f} seconds"
                )  # Display total inference time
                print(f"Min: {min(sorted_inference_times):.4f} seconds")
                print(f"Max: {max(sorted_inference_times):.4f} seconds")
                print(f"\nDetailed sorted results saved to: {result_file}")

                # Display the best result
                best_result = sorted_detailed_results[0]
                print("\nBest Answer:")
                print(
                    f"Best answer was found on page {best_result['page_number']}: {best_result['prediction']}"
                )

                return {
                    "metrics": bert_metrics,
                    "rougeL_best_prediction": rouge_l_best,
                    "inference_times": {
                        "average": avg_inference_time,
                        "total": total_inference_time,  # Include total inference time
                        "min": min(sorted_inference_times),
                        "max": max(sorted_inference_times),
                    },
                }

        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            import traceback

            print(traceback.format_exc())
            return None


def main():
    # verify connection to Ollama
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            available_models = [
                model["name"] for model in response.json().get("models", [])
            ]
            print("Available models:", available_models)
        else:
            print("Could not get model list from Ollama")
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        return

    evaluator = ContextualLLMEvaluator()

    # Create sample test data if file doesn't exist
    sample_test_data = [
        {
            "input": "What is the expected size of battery recycling in 2030?",
            "reference": "The global EV battery recycling market is projected to exceed $46.5 billion by 2030, driven by the rapid growth in electric vehicles and an increasing wave of battery retirements​",
        }
    ]

    if not os.path.exists("test_data.json"):
        with open("test_data.json", "w") as f:
            json.dump(sample_test_data, f, indent=4)

    test_data = evaluator.load_test_data("test_data.json")

    # Use the actual available models
    models = [
        "llava",  # multimodal
    ]

    # Store results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {"timestamp": timestamp, "models": {}}

    for model_name in models:
        model_results = evaluator.evaluate_model(model_name, test_data)
        if model_results:
            results["models"][model_name] = model_results

    if results["models"]:
        results_file = f"evaluation_results/combined_results_{timestamp}.json"
        os.makedirs("evaluation_results", exist_ok=True)
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\nCombined results saved to: {results_file}")


if __name__ == "__main__":
    main()
