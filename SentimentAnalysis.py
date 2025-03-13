import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import joblib

# Load the dataset
df = pd.read_csv('Sentiment_Dataset.csv')

# Question 1: Data Exploration and Dataset Overview
print("Dataset Overview:\n")
print(df.info())
print("\nSummary Statistics:\n")
print(df.describe())
print("\nFirst 5 Rows:\n")
print(df.head())

# Create a data dictionary
column_descriptions = {
    "id": "Unique identifier for each record",
    "tweet": "The actual text content of the review, tweet, or comment",
    "label": "Sentiment label (e.g., positive, negative, neutral)"
}
data_dict_df = pd.DataFrame(list(column_descriptions.items()), columns=['Column', 'Description'])
print("\nData Dictionary:\n")
print(data_dict_df)

# Question 2: Data Cleaning and Quality Assurance
print("\nChecking for Missing Values:\n")
print(df.isnull().sum())
print("\nChecking for Duplicate Records:\n")
print(df.duplicated().sum())

df = df.drop_duplicates()
df = df.dropna(subset=['tweet', 'label'])
df['id'] = df['id'].astype(int)
print("\nCleaned Dataset Overview:\n")
print(df.info())

# Question 3: Text Preprocessing and Transformation
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.strip()
    return text

df['cleaned_text'] = df['tweet'].apply(clean_text)
print("\nText Cleaning Example:\n")
print(df[['tweet', 'cleaned_text']].head())

# Question 4: Exploratory Data Analysis (EDA)
df['text_length'] = df['cleaned_text'].apply(len)
sns.histplot(df['text_length'], bins=30, kde=True)
plt.title("Distribution of Text Lengths")
plt.show()

# Question 5: Sentiment Label Distribution Analysis
sns.countplot(x='label', data=df)
plt.title("Sentiment Distribution")
plt.show()

# Question 6: Feature Engineering
df['word_count'] = df['cleaned_text'].apply(lambda x: len(x.split()))
sns.boxplot(x='label', y='word_count', data=df)
plt.title("Word Count by Sentiment")
plt.show()

# Question 7: Statistical Analysis
print("\nStatistical Summary:\n", df[['text_length', 'word_count']].describe())

# Question 8: Predictive Modeling
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nModel Performance:\n", classification_report(y_test, y_pred))

# Question 9: Hyperparameter Tuning
param_grid = {'C': [0.1, 1, 10]}
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("\nBest Parameters:", grid_search.best_params_)

# Question 10: Model Deployment
joblib.dump(grid_search.best_estimator_, 'sentiment_model.pkl')
print("Model saved as sentiment_model.pkl")
