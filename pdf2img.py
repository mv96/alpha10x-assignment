from pdf2image import convert_from_path
import os
import time
from joblib import Parallel, delayed
from tqdm import tqdm
import PyPDF2


class PDFToImageConverter:
    def __init__(self, pdf_path, output_dir, chunk_size=5):
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        os.makedirs(self.output_dir, exist_ok=True)

    def get_total_pages(self):
        """Get the total number of pages in the PDF."""
        with open(self.pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            return len(reader.pages)

    def convert_pages(self, page_range):
        """Convert a range of pages to images."""
        start_page, end_page = page_range
        return convert_from_path(
            self.pdf_path, first_page=start_page, last_page=end_page
        )

    def save_image(self, i, image):
        """Save an image to the output directory."""
        image_path = f"{self.output_dir}/page_{i + 1}.bmp"
        image.save(image_path, "BMP")

    def convert_pdf_to_images(self, n_jobs=-2):
        """Main method to convert the PDF to images."""
        start = time.time()
        try:
            total_pages = self.get_total_pages()

            # Create a list of page ranges based on the chunk size without overlap
            page_ranges = [
                (i, min(i + self.chunk_size - 1, total_pages))
                for i in range(1, total_pages + 1, self.chunk_size)
            ]

            # Use joblib's Parallel to convert pages concurrently
            with tqdm(total=len(page_ranges), desc="Converting pages") as pbar:
                images = Parallel(n_jobs=-2, backend="threading")(
                    delayed(self.convert_pages)(page_range)
                    for page_range in page_ranges
                )
                pbar.update(len(page_ranges))  # Update progress bar after conversion

            # Flatten the list of images
            images = [image for sublist in images for image in sublist]

            # Use joblib's Parallel to save images concurrently
            with tqdm(total=len(images), desc="Saving images") as pbar:
                Parallel(n_jobs=n_jobs, backend="threading")(
                    delayed(self.save_image)(i, image) for i, image in enumerate(images)
                )
                pbar.update(len(images))  # Update progress bar after saving

        except Exception as e:
            print(f"An error occurred: {e}")

        elapsed_time = time.time() - start
        print(f"Total time taken: {elapsed_time:.2f} seconds")


# Example usage
if __name__ == "__main__":
    test_pdf = "./assets/deloitte_en_lithium_pov_cn_20221114.pdf"
    output_dir = "output_images"
    chunk_size = 5  # You can change this value as needed

    converter = PDFToImageConverter(test_pdf, output_dir, chunk_size)
    converter.convert_pdf_to_images(n_jobs=-2)
