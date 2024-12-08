# MarketMind
This repository contains code for predicting stock trends using sentiment analysis of Reddit discussions. It includes scripts for scraping Reddit posts, preprocessing text, sentiment analysis, topic modeling and training a machine learning model, and a Flask web application for user interaction.

RUN THE 'scrap_reddit.py' IN THE CODE EDITOR 


# Stock Trend Prediction Using Reddit Sentiment Analysis and Topic Modelling

## Description
This repository contains the code for predicting stock trends using sentiment analysis of Reddit discussions. The main script, `scrap_reddit.py`, performs data scraping, text preprocessing, sentiment analysis, and trains a machine learning model to predict stock trends based on the Reddit posts. The `app.py` script provides a simple Flask web frontend to interact with the model, allowing users to input stock names and get trend predictions.

## Contents
- MAIN FILE  `scrap_reddit.py`: Main script for scraping Reddit data, preprocessing the text, performing sentiment analysis, training the model, and making predictions.
- OPTIONAL(FRONTEND) `app.py`: Simple Flask frontend for user interaction.
- `index2.html`: HTML file used by the Flask app for the web interface.
- `README.md`: This file.

## Setup Instructions

### Dependencies
- Python 3.x
- Flask
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
   - Alternatively, you can download the repository as a ZIP file and extract it.

2. **Install Dependencies**
   - Navigate to the project directory:
     `cd <project-directory>`
   - Install the required dependencies using pip:
     `pip install -r requirements.txt`
   - If you don't have a `requirements.txt` file, manually install the dependencies:
     `pip install flask praw nltk scikit-learn vaderSentiment yfinance pandas`
   - Download necessary NLTK data:
     ```python
     import nltk
     nltk.download('punkt')
     nltk.download('stopwords')
     nltk.download('wordnet')
     ```

### Running the Code
1. **Run the Main Script**: Execute `scrap_reddit.py` to perform data scraping and model training.
    `python scrap_reddit.py`

2. **Start the Flask Web App**: Run `app.py` to start the web application.
    `python app.py`
   - Open your web browser and navigate to `http://127.0.0.1:5000` to use the application.

### Usage
1. **Scrape Reddit Data**
   - Run `scrap_reddit.py` to collect Reddit posts related to specific stocks:
     `python scrap_reddit.py`
   - Enter the stock name (e.g., RELIANCE, TATAMOTORS) in the input field and click "Analyze Stock".

2. **Analyze Stock Trends**
   - The application will display the predicted trend, market sentiment, and classification report based on Reddit sentiment analysis.

