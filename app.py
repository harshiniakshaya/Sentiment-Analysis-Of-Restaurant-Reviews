from flask import Flask, request, render_template
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os
import pandas as pd

app = Flask(__name__)

with open('Review_model.pkl','rb') as file:
    model=pickle.load(file)

with open('CountVectorizer.pkl', 'rb') as cv_file:
    cv = pickle.load(cv_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']

        review = re.sub('[^a-zA-Z]', ' ', review)
        review = review.lower()
        review = review.split()

        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)

        review_vect = cv.transform([review]).toarray()

        prediction = model.predict(review_vect)
        sentiment = 'Positive' if prediction == 1 else 'Negative'

        output_file = 'reviews.csv'
        new_data = pd.DataFrame({'Review': [request.form['review']], 'Sentiment': [sentiment]})
        
        if os.path.isfile(output_file):
            df = pd.read_csv(output_file)
            df = pd.concat([df, new_data], ignore_index=True)
        else:
            df = new_data
        
        df.to_csv(output_file, index=False)

        return render_template('index.html', prediction=sentiment)

@app.route('/analytics')
def analytics():
    output_file = 'reviews.csv'
    if os.path.isfile(output_file):
        df = pd.read_csv(output_file)
        total_reviews = len(df)
        positive_reviews = len(df[df['Sentiment'] == 'Positive'])
        negative_reviews = len(df[df['Sentiment'] == 'Negative'])
        reviews = df.to_dict(orient='records')
    else:
        total_reviews = 0
        positive_reviews = 0
        negative_reviews = 0
        reviews = []
    
    return render_template('analytics.html', total=total_reviews, positive=positive_reviews, negative=negative_reviews, reviews=reviews)

if __name__ == '__main__':
    app.run(debug=True)

