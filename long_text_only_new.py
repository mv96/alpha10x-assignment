import evaluate
import json
import os
import numpy as np
import requests
import time
from datetime import datetime


class ContextualLLMEvaluator:
    def __init__(self):
        self.rouge = evaluate.load("rouge")
        self.bert_score = evaluate.load("bertscore")
        self.ollama_api = "http://localhost:11434/api/generate"

        # Load context from combined_output.txt
        current_dir = os.path.dirname(os.path.abspath(__file__))
        context_path = os.path.join(current_dir, "combined_output.txt")

        try:
            with open(context_path, "r", encoding="utf-8") as f:
                self.context = f.read().strip()
            print("Successfully loaded context file")
        except Exception as e:
            print(f"Error loading context file: {e}")
            self.context = ""

    def load_test_data(self, json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
            print("test_data: ", data)
            return data

    def get_model_response(self, model_name, prompt):
        """Get response from Ollama API with stream handling and timing"""
        seed = 42  # Set your desired seed for reproducibility
        data = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "temperature": 0,
            "seed": seed,  # Include the seed in the request
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

    def get_model_predictions(self, model_name, test_data):
        predictions = []
        references = []
        inference_times = []

        for item in test_data:
            try:
                print(f"\nProcessing question for {model_name}:", item["input"])

                # Create prompt with context and question
                prompt = f"""Context: {self.context} 
Question: {item['input']} 
Please provide a concise answer based on the context above.
Also, reference the pages at the beginning where you found the answer (if any).
Additionally, include a confidence score between 0 and 1 indicating how sure you are about the answer."""

                # Get prediction using Ollama API directly
                prediction, inference_time = self.get_model_response(model_name, prompt)

                if prediction:
                    predictions.append(prediction)
                    references.append(item["reference"])
                    inference_times.append(inference_time)

                    print(f"\nPrediction: {prediction}")
                    print(f"Reference: {item['reference']}")
                    print(f"Inference Time: {inference_time:.4f} seconds")
                else:
                    print("No valid response received from model")

            except Exception as e:
                print(f"Error processing question: {e}")
                import traceback

                print(traceback.format_exc())
                continue

        return predictions, references, inference_times

    def compute_metrics(self, predictions, references):
        if not predictions or not references:
            print("No predictions or references to evaluate")
            return None

        try:
            # Compute ROUGE-L scores
            rouge_scores = self.rouge.compute(
                predictions=predictions, references=references, use_stemmer=True
            )

            # Compute BERT scores with required parameters
            bert_scores = self.bert_score.compute(
                predictions=predictions,
                references=references,
                model_type="bert-base-uncased",  # Specify the language, e.g., 'en' for English
                # Alternatively, you can specify model_type if needed
                # model_type='microsoft/deberta-xlarge-mnli'
            )

            return {
                "rouge": rouge_scores["rougeL"],  # Return only ROUGE-L scores
                "bertscore": bert_scores["f1"],  # Return BERT scores
            }

        except Exception as e:
            print(f"Error computing metrics: {e}")
            return None

    def evaluate_model(self, model_name, test_data):
        print(f"\nEvaluating {model_name}...")

        try:
            # Get predictions with timing
            predictions, references, inference_times = self.get_model_predictions(
                model_name, test_data
            )

            if not predictions or not references:
                print(f"No predictions generated for {model_name}")
                return None

            # Compute metrics
            metrics = self.compute_metrics(predictions, references)

            if metrics:
                # Calculate average inference time
                avg_inference_time = sum(inference_times) / len(inference_times)

                # Save detailed results
                results_dir = "evaluation_results"
                os.makedirs(results_dir, exist_ok=True)

                # Get timestamp for the evaluation
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                detailed_results = {
                    "model_name": model_name,
                    "timestamp": timestamp,
                    "predictions": predictions,
                    "references": references,
                    "metrics": metrics,
                    "inference_times": {
                        "individual": inference_times,
                        "average": avg_inference_time,
                        "min": min(inference_times),
                        "max": max(inference_times),
                    },
                }

                safe_model_name = model_name.replace(":", "_").replace("/", "_")
                result_file = (
                    f"{results_dir}/{safe_model_name}_{timestamp}_results.json"
                )
                with open(result_file, "w") as f:
                    json.dump(detailed_results, f, indent=4)

                print(f"\n{model_name} Results:")
                print("Predictions:", predictions)
                print("\nPerformance Metrics:")
                print("ROUGE-L Scores:", metrics["rouge"])  # Print ROUGE-L scores
                print("BERT Scores:", metrics["bertscore"])  # Print BERT scores
                print("\nInference Time Statistics:")
                print(f"Average: {avg_inference_time:.4f} seconds")
                print(f"Min: {min(inference_times):.4f} seconds")
                print(f"Max: {max(inference_times):.4f} seconds")
                print(f"\nDetailed results saved to: {result_file}")

                return {
                    "metrics": metrics,
                    "inference_times": {
                        "average": avg_inference_time,
                        "min": min(inference_times),
                        "max": max(inference_times),
                    },
                }

        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            import traceback

            print(traceback.format_exc())
            return None


def main():
    # First, let's verify we can connect to Ollama
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

    test_data = evaluator.load_test_data("question_set.json")

    # Use the actual available models
    models = ["llama3.2:1b", "mistral:latest"]

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
