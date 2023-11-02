# News Text Classification Using BERT

BERT is a state-of-the-art transformer model used to perform complex NLP tasks like NLU, NLG, Sentiment Analysis, Text Classification, and it delivers very high-performance metrics,sometimes even surpassing human capabilities.

---

## Objective

To perform text classification on the AG News dataset using the state-of-the-art transformer model BERT.

---

## Data Description

G News (AG’s News Corpus) is a sub dataset of AG's corpus of news articles constructed by assembling titles and description fields of articles from the four largest classes: "World," "Sports," "Business," and "Sci/Tech" of AG’s Corpus.

---

## Installation

1. Install the required packages stated in the `requirements.txt` file.

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

2. The entire repository is modularized into individual sections that perform specific tasks. First, navigate to the `src` folder.

   Under `src`, there are two primary packages:
   - `ML_Pipeline`: This contains individual modules with different function declarations to perform specific Machine Learning tasks.
   - `engine.py`: This is the heart of the project, as all the function calls are made here.

1. Run/Debug the `engine.py` file, and all the necessary steps will be automatically taken care of according to the logic.

2. All input datasets are stored in the `input` folder.

3. All predictions and models are stored in the `output` folder.

---