import os
import time
from PIL import Image
import pytesseract
from joblib import Parallel, delayed
from tqdm import tqdm


class ImageOCRConverter:
    def __init__(self, image_folder, output_folder, expected_formats=None, n_jobs=-2):
        self.image_folder = image_folder
        self.output_folder = output_folder
        self.expected_formats = expected_formats or [
            ".png",
            ".jpg",
            ".jpeg",
            ".tiff",
            ".bmp",
            ".gif",
        ]
        self.n_jobs = n_jobs
        os.makedirs(self.output_folder, exist_ok=True)

    def get_image_files(self):
        """Get a list of image files in the specified folder."""
        return [
            filename
            for filename in os.listdir(self.image_folder)
            if filename.lower().endswith(tuple(self.expected_formats))
        ]

    def process_image(self, filename):
        """Perform OCR on a single image and save the extracted text."""
        file_path = os.path.join(self.image_folder, filename)
        try:
            # Open the image
            image = Image.open(file_path)

            # Perform OCR
            text = pytesseract.image_to_string(image)

            # Save the extracted text to a file
            output_file = os.path.join(
                self.output_folder, f"{os.path.splitext(filename)[0]}.txt"
            )
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(text)

            return
        except Exception as e:
            return f"Failed to process {filename}: {e}"

    def convert_images_to_text(self):
        """Main method to convert images to text using OCR."""
        start_time = time.time()
        image_files = self.get_image_files()

        # Use joblib's Parallel to process images concurrently
        with tqdm(total=len(image_files), desc="Processing images") as pbar:
            results = Parallel(n_jobs=self.n_jobs, backend="multiprocessing")(
                delayed(self.process_image)(filename) for filename in image_files
            )
            pbar.update(len(image_files))  # Update progress bar after processing

        # Print results
        for result in results:
            print(result)

        elapsed_time = time.time() - start_time
        print(f"Total time taken: {elapsed_time:.2f} seconds")


# Example usage
if __name__ == "__main__":
    image_folder = "./output_images"
    output_folder = "./ocr_output"
    expected_formats = [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"]
    n_jobs = -2  # You can change this value as needed

    converter = ImageOCRConverter(image_folder, output_folder, expected_formats, n_jobs)
    converter.convert_images_to_text()
