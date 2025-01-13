import logging
from flask import Flask, render_template, request
import praw
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import LatentDirichletAllocation
import yfinance as yf
from datetime import date, timedelta
import re

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Reddit API client
reddit = praw.Reddit(
    client_id='bDWJ0CETEqdyBi1XhH3gGA',
    client_secret='YpLczCPqjfsCKRk8RlcKbf9cpOYpuA',
    user_agent='MarketMind'
)

# Initialize Flask app
app = Flask(__name__)

# StockTrendPredictor Class Definition
class StockTrendPredictor:
    def __init__(self, reddit_client):
        """Initialize with Reddit client."""
        self.reddit = reddit_client
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = CountVectorizer(stop_words='english')
        self.model = LogisticRegression()
        self.analyzer = SentimentIntensityAnalyzer()

    def scrape_reddit_data(self, stock_name, limit=300):
        """Scrape Reddit posts from relevant subreddits."""
        subreddits = ['wallstreetbets', 'stocks', 'investing', 'IndianStockMarket', 'IndiaInvestments']
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

    def preprocess_text(self, text):
        """Preprocess the Reddit text by cleaning and tokenizing."""
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'[^A-Za-z0-9\s]', ' ', text)  # Remove non-alphanumeric characters
        text = text.lower()  # Convert text to lowercase
        tokens = word_tokenize(text)  # Tokenize text
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]
        return ' '.join(tokens)

    def prepare_data(self, posts):
        """Prepare data by preprocessing and vectorizing."""
        clean_posts = [self.preprocess_text(post) for post in posts]
        X = self.vectorizer.fit_transform(clean_posts)
        return X

    def analyze_sentiment(self, posts):
        """Analyze sentiment of the posts using VADER."""
        sentiments = [self.analyzer.polarity_scores(post)['compound'] for post in posts]
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        return sentiments, avg_sentiment

    def get_stock_data(self, stock_name, start_date, end_date):
        """Fetch historical stock data from Yahoo Finance."""
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

    def topic_modeling(self, posts):
        """Perform topic modeling using LDA."""
        logging.info("Performing topic modeling...")
        clean_posts = [self.preprocess_text(post) for post in posts]
        X = self.vectorizer.fit_transform(clean_posts)
        lda = LatentDirichletAllocation(n_components=5, random_state=42)
        lda.fit(X)
        feature_names = self.vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            topic_words = " ".join([feature_names[i] for i in topic.argsort()[:-11:-1]])
            topics.append(f"Topic #{topic_idx + 1}: {topic_words}")
        return topics

    def train_model(self, X, y):
        """Train the logistic regression model."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred)
        return report

    def predict_stock_trend(self, stock_name):
        """Predict stock trend based on Reddit sentiment."""
        end_date = date.today()
        start_date = end_date - timedelta(days=365)
        stock_data = self.get_stock_data(stock_name, start_date, end_date)
        if stock_data is None:
            return None, None, None, None
        posts = self.scrape_reddit_data(stock_name)
        if not posts:
            return None, None, None, None
        X = self.prepare_data(posts)
        topics = self.topic_modeling(posts)
        sentiments, avg_sentiment = self.analyze_sentiment(posts)
        y = [1 if sentiment > 0 else 0 for sentiment in sentiments]
        if len(set(y)) < 2:
            return None, None, None, None
        report = self.train_model(X, y)

        # Determine the market sentiment based on the average sentiment score
        market_sentiment = 'Buy' if avg_sentiment > 0 else 'Sell'
        sentiment_label = 'Uptrend' if avg_sentiment > 0 else 'Downtrend'
        
        return sentiment_label, report, topics, market_sentiment


# Flask routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze_stock():
    stock_name = request.form.get("stock_name")
    if not stock_name:
        return "Please provide a stock name.", 400

    predictor = StockTrendPredictor(reddit)
    sentiment_label, report, topics, market_sentiment = predictor.predict_stock_trend(stock_name)

    if sentiment_label is None:
        return render_template(
            "index.html",
            error="Unable to predict stock trend. Please try a different stock or check your data."
        )

    return render_template(
        "index.html",
        stock_name=stock_name,
        sentiment_label=sentiment_label,
        classification_report=report,
        topics=topics,
        market_sentiment=market_sentiment
    )


if __name__ == "__main__":
    app.run(debug=True)

