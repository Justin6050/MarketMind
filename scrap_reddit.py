import praw
import re
import nltk
import pandas as pd
from datetime import date, timedelta
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
import logging
from concurrent.futures import ThreadPoolExecutor
from sklearn.decomposition import LatentDirichletAllocation

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
                                    # INITIALIZATION

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize PRAW client (Reddit API)
reddit= praw.Reddit(client_id='bDWJ0CETEqdyBi1XhH3gGA',
            client_secret='YpLczCPqjfsCKRk8RlcKbf9cpOYpuA',
            user_agent='MarketMind')  

class StockTrendPredictor:
    def __init__(self, reddit_client):
        self.reddit = reddit_client
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = CountVectorizer(stop_words='english')
        self.model = LogisticRegression()
        self.analyzer = SentimentIntensityAnalyzer()


                                     # DATA SCRAPING
                # Function to collect Reddit posts related to a specific stock


    def scrape_reddit_data(self, stock_name, limit=300):
        """Scrape Reddit posts from relevant subreddits"""
        subreddits = [ 'IndianStockMarket', 'IndiaInvestments', 'bombaystockexchange', 'mutualfunds',
                      'IndianStreetBets', 'ShareMarketupdates','stocks', 'investing',
                      'wallstreetbets','NSEbets']
        posts = []
        for subreddit in subreddits:
            try:
                logging.info(f"Scraping posts from /r/{subreddit} for {stock_name}...")
                subreddit_instance = self.reddit.subreddit(subreddit)
                for post in subreddit_instance.search(stock_name, sort='new', limit=limit):
                    posts.append(post.title + " " + post.selftext)
            except Exception as e:
                logging.error(f"Error scraping Reddit data from /r/{subreddit}: {e}")
        return posts
     
       # Using ThreadPoolExecutor for concurrent scraping

        with ThreadPoolExecutor() as executor:
            results = executor.map(scrape_subreddit, subreddits)
            for result in results:
                posts.extend(result)
        return posts

                                       # TEXT PREPROCESSING
                     # Cleaning, tokenizing, and lemmatizing Reddit posts

    
    def preprocess_text(self, text):
        """Preprocess the Reddit text by cleaning and tokenizing"""
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'[^A-Za-z0-9\s]', ' ', text)  # Remove non-alphanumeric characters
        text = text.lower()  # Convert text to lowercase
        tokens = word_tokenize(text)  # Tokenize text
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]  # Lemmatize and remove stopwords
        return ' '.join(tokens)

 
                                        # DATA PREPARATION
                     # Converting the cleaned Reddit posts into vector format for model training

   
    def prepare_data(self, posts):
        """Prepare data by preprocessing and vectorizing"""
        clean_posts = [self.preprocess_text(post) for post in posts]
        X = self.vectorizer.fit_transform(clean_posts)  # Convert text to vector form
        return X
    

                                          # FETCHING STOCKS DATA
                           # Retrieving historical stock data from Yahoo Finance

   
    def get_stock_data(self, stock_name, start_date, end_date):
        """Fetch historical stock data from Yahoo Finance"""
        try:
            ticker = f"{stock_name}.NS"
            stock_data = yf.download(ticker, start=start_date, end=end_date)
            if stock_data.empty:
                logging.warning(f"No historical data found for the stock '{stock_name}'!")
                return None
            return stock_data
        except Exception as e:
            logging.error(f"Error fetching stock data: {e}")
            return None
        
        
                                          # SENTIMENT ANALYSIS
                    # Use VADER sentiment analysis to evaluate sentiment of Reddit posts

    def analyze_sentiment(self, posts):
        """Analyze sentiment of the posts using VADER"""
        sentiments = [self.analyzer.polarity_scores(post)['compound'] for post in posts]
        avg_sentiment = sum(sentiments) / len(sentiments)
        return sentiments, avg_sentiment


                                             # MODEL TRAINING
                  # Training a logistic regression model to predict stock trends based on sentiment
    
   
    def train_model(self, X, y):
        #Train the logistic regression model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        logging.info(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

        logging.info(f"Precision: {precision_score(y_test, y_pred):.2f}")
        logging.info(f"Recall: {recall_score(y_test, y_pred):.2f}")
        logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
        return self.model


                                                # TOPIC MODELING
                          # Apply LDA (Latent Dirichlet Allocation) to identify topics in Reddit posts

  
    def topic_modeling(self, posts):
        """Perform topic modeling using LDA"""
        logging.info("Performing topic modeling...")
        clean_posts = [self.preprocess_text(post) for post in posts]
        X = self.vectorizer.fit_transform(clean_posts)

        # Applying LDA for topic modeling
        lda = LatentDirichletAllocation(n_components=5, random_state=42)  # Set the number of topics
        lda.fit(X)
        
        feature_names = self.vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(lda.components_):
            logging.info(f"Topic #{topic_idx}:")
            print(" ".join([feature_names[i] for i in topic.argsort()[:-11:-1]]))  # Print top 10 words for each topic


                                          # STOCK TREND PREDICTION
                               #Predicting stock trend (0 for down, 1 for up)


    def predict_stock_trend(self, stock_name):
        
        # Defining the time period for fetching historical data
        end_date = date.today()
        start_date = end_date - timedelta(days=365)  # Past one year

        # Fetching historical stock data of specifically one year
        logging.info(f"Fetching historical data for {stock_name} from {start_date} to {end_date}...")
        stock_data = self.get_stock_data(stock_name, start_date, end_date)
        if stock_data is None:
            return

        # Scrape and process Reddit data
        posts = self.scrape_reddit_data(stock_name)
        if not posts:
            logging.warning("No data found for the stock!")
            return

        # Preprocess and prepare the data
        X = self.prepare_data(posts)

        # Perform topic modeling
        self.topic_modeling(posts)

        # Perform sentiment analysis
        sentiments, avg_sentiment = self.analyze_sentiment(posts)
        sentiment_label = 1 if avg_sentiment > 0 else 0  # Positive sentiment for uptrend, negative for downtrend

        # Create target labels based on sentiment analysis results
        y = [1 if sentiment > 0 else 0 for sentiment in sentiments]

        # Ensure at least two classes are present
        if len(set(y)) < 2:
            logging.warning("Insufficient class variety in data for training.")
            return

        # Train the model
        self.train_model(X, y)

        # Make prediction for stock trend based on sentiment
        test_data = self.vectorizer.transform([self.preprocess_text(posts[-1])])
        prediction = self.model.predict(test_data)
        sentiment = 'buy' if avg_sentiment > 0 else 'sell'
        
        logging.info(f"Predicted trend for {stock_name}: {'Uptrend' if prediction[0] == 1 else 'Downtrend'}")
        logging.info(f"Market sentiment is towards: {sentiment}")



def main():
    stock_name = input("Enter the stock name (e.g., RELIANCE, TATAMOTORS, INFY): ")
    predictor = StockTrendPredictor(reddit)
    predictor.predict_stock_trend(stock_name)

if __name__ == "__main__":
    main()
