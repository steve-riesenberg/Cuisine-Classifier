import flask
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import re

app = flask.Flask(__name__, template_folder = 'templates')

with open('model/cuisine_df.pkl', 'rb') as file:
    df = pickle.load(file)

with open('model/clf.pkl', 'rb') as file:
    clf = pickle.load(file)

le = preprocessing.LabelEncoder()
le.fit(df.cuisine)
X_labels = le.transform(df.cuisine)

vectorizer = CountVectorizer(tokenizer=lambda x: [i.strip() for i in x.split(',')])
vectorizer.fit(df.ingredients.apply(','.join), X_labels)

pipeline = Pipeline([('vectorizer', vectorizer), ('clf', clf)])

@app.route('/', methods = ['GET', 'POST'])

def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':
        ingredients = flask.request.form['ingredients']
        ingredients = ingredients.splitlines()
        ingredients = ','.join(ingredients)
        ingredients = re.sub(r'[0-9]+', '', ingredients)
        print(ingredients)
        prediction = pipeline.predict([ingredients])
        result = le.inverse_transform(prediction)[0]
        return flask.render_template('main.html', original_input = {'ingredients':ingredients}, result = result)


if __name__ == '__main__':
    app.run(debug = True)
