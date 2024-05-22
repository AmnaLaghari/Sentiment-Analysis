import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


from utils import clean_data, create_sentiment

# step1: reading data set
df = pd.read_csv('tripadvisor_hotel_reviews.csv')

# step2: data preprocessing
df['Sentiment'] = df['Rating'].apply(create_sentiment)
# print(df.head())

df['Review'] = df['Review'].apply(clean_data)

# Step 4: TF-IDF Transformation (converting words into vectors)
tfidf = TfidfVectorizer(strip_accents=None, 
                        lowercase=False,
                        preprocessor=None)

X = tfidf.fit_transform(df['Review'])

# Step 5: Building and Evaluating the Machine Learning Model
y = df['Sentiment'] # target variable
X_train, X_test, y_train, y_test = train_test_split(X,y)

lr = LogisticRegression(solver='liblinear')
lr.fit(X_train,y_train) # fit the model
preds = lr.predict(X_test) # make predictions
print(preds)

#evaluate performance
accuracy_score(preds,y_test)
print(accuracy_score.__dict__)