# 📄 PDF Question Answering Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Transformers](https://img.shields.io/badge/Transformers-4.0%2B-orange.svg)
![Torch](https://img.shields.io/badge/PyTorch-1.7%2B-red.svg)
![Tesseract](https://img.shields.io/badge/Tesseract-4.0%2B-green.svg)
![Evaluate](https://img.shields.io/badge/Evaluate-0.2%2B-yellow.svg)
![Joblib](https://img.shields.io/badge/Joblib-1.0%2B-lightgrey.svg)
![Pillow](https://img.shields.io/badge/Pillow-8.0%2B-purple.svg)

Welcome to the **PDF Question Answering Pipeline**! 🚀 This project provides a comprehensive solution to convert PDF documents into text, perform Optical Character Recognition (OCR), and evaluate the extracted text using various advanced models for question answering tasks. Whether you're dealing with large-scale document processing or need robust text analysis, this pipeline has got you covered! 🛠️

## 📜 Features

- **📚 PDF to Image Conversion**: Efficiently convert multi-page PDF files into high-quality images.
- **📝 Image to Text (OCR)**: Extract text from images using state-of-the-art OCR techniques.
- **🔗 Text Combination**: Merge extracted text files into a single, cohesive document for easier analysis.
- **🧠 Model Evaluation for QA**: Assess the quality of the extracted text using multiple evaluation models specifically designed for question answering tasks.
- **⚡ Parallel Processing**: Utilize multi-threading and multi-processing to speed up conversions and evaluations.
- **📊 Detailed Metrics**: Generate comprehensive metrics to evaluate model performance in question answering.

## 📊 Results Summary

| Model Name                 | ROUGE-L Score | BERTScore F1 | Inference Time (s) | Comments                           |
| -------------------------- | ------------- | ------------ | ------------------ | ---------------------------------- |
| Contextual LLM Evaluator   | 0.85          | 0.90         | 1.23               | High accuracy                      |
| LayoutLMv3 Evaluator       | 0.80          | 0.88         | 1.45               | Good for layout-based tasks        |
| Document Evaluator (UDOP)  | 0.82          | 0.87         | 1.30               | Effective for structured documents |
| Document Evaluator (Donut) | 0.78          | 0.85         | 1.50               | Works well with images             |
| Short Text Model Evaluator | 0.75          | 0.80         | 1.10               | Suitable for short queries         |

_Note: The scores and times are illustrative. Please replace them with actual results from your evaluations._

## 📦 Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/pdf-processing-pipeline.git
   cd pdf-processing-pipeline
   ```

2. **Install Dependencies**

   Ensure you have Python 3.x installed. Then, install the required Python libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. **Additional Setup**

   - **Tesseract OCR**: Make sure Tesseract is installed on your system. You can download it from [here](https://github.com/tesseract-ocr/tesseract).
   - **Ollama API**: Ensure the Ollama API is running locally if you're using models that rely on it.

## 🛠️ Usage

Follow these steps to process your PDF documents and evaluate the extracted text for question answering:

1. **Convert PDF to Images**

   ```bash
   python pdf2img.py
   ```

   This will convert your PDF files into images and save them in the `output_images` directory.

2. **Perform OCR on Images**

   ```bash
   python img2ocr.py
   ```

   This step extracts text from the images and saves the output in the `ocr_output` directory.

3. **Combine Extracted Texts**

   ```bash
   python combine_texts.py
   ```

   Merges all text files into a single `combined_output.txt` for streamlined analysis.

4. **Evaluate with Models for QA**

   Choose the evaluator based on your requirements:

   - **Contextual LLM Evaluator**
     ```bash
     python long_text_only_new.py
     ```
   - **LayoutLMv3 Evaluator**
     ```bash
     python layoutlm.py
     ```
   - **Document Evaluator (UDOP and Donut)**
     ```bash
     python udop.py
     python donut.py
     ```
   - **Short Text Only Model Evaluator**
     ```bash
     python short_text_only_model.py
     ```

## 📝 Requirements

- **Programming Language**: Python 3.x
- **Libraries**:
  - `pdf2image`
  - `pytesseract`
  - `transformers`
  - `evaluate`
  - `natsort`
  - `joblib`
  - `tqdm`
  - `Pillow`
  - `requests`
  - `numpy`
  - `torch`
- **Other Tools**:
  - Tesseract OCR
  - Ollama API (if using related evaluators)

## 📂 Project Structure

```
pdf-processing-pipeline/
├── pdf2img.py
├── img2ocr.py
├── combine_texts.py
├── long_text_only_new.py
├── layoutlm.py
├── udop.py
├── donut.py
├── short_text_only_model.py
├── requirements.txt
├── README.md
└── LICENSE
```

## 📝 License

This project is licensed under the following terms:

- You are allowed to use this code **solely for academic purposes**, specifically for **assignment evaluations**.
- **No part of this code may be used in any commercial environment or for profit**.
- **No modifications, adaptations, or redistributions of this code are permitted**.
- You must not use this code for any other purpose, including but not limited to personal projects, commercial applications, or any form of distribution.

By using this code, you agree to these terms.

## 📞 Contact Information

**Shrey Mishra**  
PhD Candidate at ENS PARIS  
Email: [mishra@di.ens.fr](mailto:mishra@di.ens.fr)

---

✨ **Happy Processing!** If you have any questions or need further assistance, feel free to open an issue or reach out. We're here to help! 😊