import os
from natsort import natsorted  # Import natsorted
import time  # Import time module


class TextCombiner:
    def __init__(self, ocr_text_path, output_file_path):
        self.ocr_text_path = ocr_text_path
        self.output_file_path = output_file_path

    def combine_texts(self):
        """Read all files in the specified directory and combine them."""
        start_time = time.time()  # Start timing
        with open(self.output_file_path, "w") as outfile:
            for index, filename in enumerate(
                natsorted(os.listdir(self.ocr_text_path))
            ):  # Use natsorted for natural sorting
                file_path = os.path.join(
                    self.ocr_text_path, filename
                )  # Define file_path here
                if os.path.isfile(file_path):  # Check if it's a file
                    # Add header before each page
                    outfile.write(
                        f"==HEADER PAGE {index + 1}==\n"
                    )  # Add header for each page
                    with open(file_path, "r") as infile:
                        outfile.write(
                            infile.read() + "\n"
                        )  # Write file content to output file
        elapsed_time = time.time() - start_time  # Calculate elapsed time
        print(
            f"Total time taken to combine texts: {elapsed_time:.2f} seconds"
        )  # Print elapsed time


# Example usage
if __name__ == "__main__":
    ocr_text_path = "./ocr_output"
    output_file_path = "./combined_output.txt"

    combiner = TextCombiner(ocr_text_path, output_file_path)
    combiner.combine_texts()
