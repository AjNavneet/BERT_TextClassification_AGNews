# BERT-Powered News Text Classification

BERT, a cutting-edge transformer model, has proven its prowess in complex NLP tasks, such as NLU, NLG, Sentiment Analysis, and Text Classification. Its exceptional performance often surpasses human capabilities.

---

## Objective

Conduct text classification on the AG News dataset using the state-of-the-art transformer model, BERT.

---

## Data Description

The AG News dataset (AG’s News Corpus) is a subset of AG's corpus, constructed by combining titles and description fields from articles in the four major classes: "World," "Sports," "Business," and "Sci/Tech" of AG’s Corpus.

---

## Installation

1. Install the required packages specified in the `requirements.txt` file.

   For Anaconda:
   ```bash
   conda create --name <yourenvname>
   conda activate <yourenvname>
   pip install -r requirements.txt
   ```

   For Python Interpreter:
   ```bash
   pip install -r requirements.txt
   ```

2. The entire repository is modularized into distinct sections, each handling specific tasks. Start by navigating to the `src` folder.

   Within `src`:
   - `ML_Pipeline`: Contains modules with function declarations for various Machine Learning tasks.
   - `engine.py`: The core of the project, orchestrating all function calls.

3. Run/Debug the `engine.py` file, and the necessary steps will be automated according to the defined logic.

4. Input datasets are stored in the `input` folder.

5. Predictions and models are saved in the `output` folder.

---
