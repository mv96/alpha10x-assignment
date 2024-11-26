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
from sentence_transformers import SentenceTransformer, util


class ContextualLLMEvaluator:
    def __init__(self):
        self.rouge = evaluate.load("rouge")
        self.bert_score = evaluate.load("bertscore")
        self.ollama_api = "http://localhost:11434/api/generate"
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )  # Load a pre-trained model for embeddings

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
            "temperature": 0,
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
        """Get response from Ollama API for image input."""
        img_base64 = self.encode_image_to_base64(image_path)
        if not img_base64:
            return None, 0

        improved_prompt = f"""
{prompt}\n
Please provide a concise and accurate answer to the question based on the provided OCR text and image.
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
            # print(f"Model response: {response_json}")  # Debugging line

            if "response" in response_json:
                response_text = response_json["response"].strip()
                return (
                    response_text,
                    inference_time,
                )  # Return only prediction and inference time

            else:
                print(f"Unexpected response format: {response_json}")
                return None, inference_time

        except Exception as e:
            print(f"Error getting model response for image: {e}")
            return None, 0

    def calculate_relevance(self, ocr_text, question):
        """Calculate relevance score based on OCR text and question using cosine similarity."""
        # Encode the texts to get their embeddings
        embeddings = self.model.encode([ocr_text, question], convert_to_tensor=True)

        # Compute cosine similarity
        similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])

        return similarity.item()  # Return the similarity score as a float

    def get_model_predictions(self, model_name, test_data):
        predictions = []
        references = []
        inference_times = []
        relevance_scores = []  # Store relevance scores

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

                        # Calculate relevance score for the current page
                        relevance_score = self.calculate_relevance(
                            context, item["input"]
                        )
                        

                        # Only proceed if the relevance score is above a certain threshold
                        if relevance_score > 0.5:  # Adjust threshold as needed
                            # Create prompt with context and question
                            prompt = f"""OCR scanned text: {context} 
                                        Question: {item['input']}"""

                            # Get prediction using vision model
                            prediction, inference_time = (
                                self.get_model_response_with_image(
                                    model_name, prompt, image_path
                                )
                            )

                            if prediction:
                                relevance_scores.append(relevance_score)
                                predictions.append(prediction)
                                references.append(item["reference"])
                                inference_times.append(inference_time)

                                print(f"\nImage: {image_path}")
                                print(f"Prediction: {prediction}")
                                print(f"Inference Time: {inference_time:.4f} seconds")
                                print(
                                    f"Relevance Score: {relevance_score:.4f}"
                                )  # Print relevance score
                            else:
                                print(
                                    f"No valid response received from model for image {image_path}. Check model availability."
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
            relevance_scores,  # Return relevance scores
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

        try:
            # Get predictions with timing
            predictions, references, inference_times, relevance_scores = self.get_model_predictions(model_name, test_data)

            # Debug prints
            print(f"Number of predictions: {len(predictions)}")
            print(f"Number of references: {len(references)}")
            print(f"Number of relevance scores: {len(relevance_scores)}")

            # Check if we have any valid predictions
            if not predictions or not references or not relevance_scores:
                print(f"No valid predictions generated for {model_name}")
                return None

            # Ensure all lists have the same length
            # if len(predictions) != len(references) or len(predictions) != len(relevance_scores):
            #     print(f"Mismatched lengths: predictions={len(predictions)}, references={len(references)}, relevance_scores={len(relevance_scores)}")
            #     return None

            # Find the best relevance score and corresponding prediction
            if len(relevance_scores) > 0:
                best_index = int(np.argmax(relevance_scores))
                best_prediction = predictions[best_index]
                best_reference = references[best_index]
                best_relevance_score = relevance_scores[best_index]
            else:
                print(f"No relevance scores available for {model_name}")
                return None

            # Compute ROUGE-L score on the best prediction only
            rouge_scores_best = self.rouge.compute(
                predictions=[best_prediction],
                references=[best_reference]
            )
            
            # Access ROUGE-L score with error handling
            rouge_l_best = rouge_scores_best.get('rougeL', 0.0)

            # Calculate BERTScore for the best prediction
            bert_scores_best = self.bert_score.compute(
                predictions=[best_prediction],
                references=[best_reference],
                model_type="bert-base-uncased",
            )
            avg_bert_f1_best = float(np.mean(bert_scores_best["f1"]))

            # Calculate average inference time
            avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0

            # Save detailed results
            results_dir = "evaluation_results"
            os.makedirs(results_dir, exist_ok=True)

            # Get timestamp for the evaluation
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            detailed_results = {
                "model_name": model_name,
                "parameter_count_billion": param_count,
                "timestamp": timestamp,
                "best_prediction": best_prediction,
                "best_reference": best_reference,
                "best_relevance_score": best_relevance_score,
                "rougeL_best_prediction": rouge_l_best,
                "bert_f1_best_prediction": avg_bert_f1_best,
                "inference_times": {
                    "individual": inference_times,
                    "average": avg_inference_time,
                    "min": min(inference_times) if inference_times else 0,
                    "max": max(inference_times) if inference_times else 0,
                },
            }

            # Save results to file
            safe_model_name = model_name.replace(":", "_").replace("/", "_")
            result_file = f"{results_dir}/{safe_model_name}_{timestamp}_results.json"
            with open(result_file, "w") as f:
                json.dump(detailed_results, f, indent=4)

            # Print results
            print(f"\nBest Answer:")
            print(f"Best prediction: {best_prediction}")
            print(f"Reference: {best_reference}")
            print(f"Relevance score: {best_relevance_score:.4f}")
            print(f"BERTScore: {avg_bert_f1_best:.4f}")
            print(f"ROUGE-L: {rouge_l_best:.4f}")
            print(f"Average inference time: {avg_inference_time:.4f} seconds")
            print(f"\nDetailed results saved to: {result_file}")

            return detailed_results

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
        "llama3.2-vision",  # multimodal
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
