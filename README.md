# MarketMind
This repository contains code for predicting stock trends using sentiment analysis of Reddit discussions. It includes scripts for scraping Reddit posts, preprocessing text, sentiment analysis, topic modeling, and training a machine learning model.

RUN THE 'scrap_reddit.py' IN THE CODE EDITOR 

# Stock Trend Prediction Using Reddit Sentiment Analysis

## Description
This repository contains the code for predicting stock trends using sentiment analysis of Reddit discussions. The main script, `scrap_reddit.py`, performs data scraping, text preprocessing, sentiment analysis, and model training. The script collects posts from relevant Reddit subreddits, processes the text data, and predicts stock trends based on sentiment analysis.

## Contents
- `scrap_reddit.py`: Main script for scraping Reddit data, preprocessing the text, performing sentiment analysis, training the machine learning model, and making predictions.
- `README.md`: This file.

## Setup Instructions

### Dependencies
- Python 3.x
- PRAW (Python Reddit API Wrapper)
- NLTK (Natural Language Toolkit)
- Scikit-learn
- VADER Sentiment Analysis
- yfinance
- Logging
- Concurrent.futures
- Pandas

### Installation
1. **Clone the Repository**
   - Clone the repository to your local machine using the following command:
     `git clone <repository-url>`
   - Alternatively, download the repository as a ZIP file and extract it.

2. **Install Dependencies**
   - Navigate to the project directory:
     `cd <project-directory>`
   - Install the required dependencies using pip:
     `pip install -r requirements.txt`
   - If you don't have a `requirements.txt` file, manually install the dependencies:
     `pip install praw nltk scikit-learn vaderSentiment yfinance pandas`
   - Download necessary NLTK data:
     ```python
     import nltk
     nltk.download('punkt')
     nltk.download('stopwords')
     nltk.download('wordnet')
     ```

### Running the Code
1. **Run the Main Script**: Execute `scrap_reddit.py` to perform data scraping, sentiment analysis, and model training.
    `python scrap_reddit.py`

### Usage
1. **Scrape Reddit Data**: The script will collect Reddit posts related to a specific stock. Enter the stock name (e.g., RELIANCE, TATAMOTORS) when prompted, and the script will fetch the data.

2. **Analyze Stock Trends**: The script will display the predicted trend (uptrend or downtrend) based on Reddit sentiment analysis, along with the classification metrics (accuracy, precision, recall).

## License
This project is licensed under the MIT License.
