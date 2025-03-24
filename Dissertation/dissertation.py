'''
    Liya Kuncheva
    aca20lk
    University of Sheffield
    Irony Detection in English Tweets
    10/05/2023
'''

'''
    To run this code on your machine, you have to install the packages from the requirements.txt file 
    by running the command
        pip install -r requirements.txt
    on the command line.
    
    The path to the data sets needs to be replaced as well.
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import nltk
from textblob import TextBlob
import numpy as np


nltk.download('punkt')
nltk.download('wordnet')


def preprocess(text):

    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove mentions
    text = re.sub(r'@\w+', '', text)

    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Initialize stemmer
    stemmer = SnowballStemmer('english')
    tokens = [stemmer.stem(word) for word in tokens]

    # Join the tokens back into a string
    preprocessed_text = ' '.join(tokens)

    '''
    # Sentiment feature extraction
    blob = TextBlob(preprocessed_text)
    sentiment_polarity = blob.sentiment.polarity
    sentiment_subjectivity = blob.sentiment.subjectivity

    # Append sentiment features to the preprocessed text
    preprocessed_text += f' {sentiment_polarity} {sentiment_subjectivity}'
    '''
    return preprocessed_text




# Main
if __name__ == '__main__':
    # Load the SemEval 2018 Task 3 dataset
    train_data = pd.read_csv("C:/data/datasets/train/SemEval2018-T3-train-taskA_emoji.txt",
                             skiprows=[0], sep="\t", header=None, names=["id", "label", "text"])
    test_data = pd.read_csv("C:/data/datasets/goldtest_TaskA/SemEval2018-T3_gold_test_taskA_emoji.txt",
                            skiprows=[0], sep="\t", header=None, names=["id", "label", "text"])


    # Split the dataset into training and testing sets and apply preprocessing
    #X_train, y_train = train_data['text'].apply(preprocess), train_data["label"]
    #X_test, y_test = test_data['text'].apply(preprocess), test_data["label"]

    # Split the dataset into training and testing sets without applying preprocessing
    X_train, y_train = train_data['text'], train_data["label"]
    X_test, y_test = test_data['text'], test_data["label"]

    # Define a pipeline for feature extraction
    text_pipeline = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 3), analyzer='word', tokenizer=word_tokenize)),
        ('tfidf', TfidfTransformer())
    ])

    X_train_vectorized = text_pipeline.fit_transform(X_train)
    X_test_vectorized = text_pipeline.transform(X_test)

    # Define models
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel="rbf", C=10, random_state=42, probability=True),
        'Logistic Regression': LogisticRegression(),
        'Naive Bayes': MultinomialNB(),
        'MLPClassifier': MLPClassifier(hidden_layer_sizes=(50, 20), max_iter=1000, random_state=42)
    }

    train_meta = np.zeros((X_train_vectorized.shape[0], len(models)))
    test_meta = np.zeros((X_test_vectorized.shape[0], len(models)))

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Train and evaluate base models
    print("Base Models:\n")
    for i, (model_name, model) in enumerate(models.items()):
        model.fit(X_train_vectorized, y_train)
        train_meta[:, i] = model.predict(X_train_vectorized)
        '''
        # This code can be used for the kfold by removing the line before
        for train_idx, val_idx in kfold.split(X_train_vectorized, y_train):
            X_train_fold, X_val_fold = X_train_vectorized[train_idx], X_train_vectorized[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
            model.fit(X_train_fold, y_train_fold)

            train_meta[val_idx, i] = model.predict(X_val_fold)
        '''
        test_meta[:, i] = model.predict(X_test_vectorized)
        y_pred = model.predict(X_test_vectorized)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=1)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=1)
        report = classification_report(y_test, y_pred)
        cmatrix = confusion_matrix(y_test, y_pred)
        print(f'{model_name}:\n {report}\n Accuracy: {accuracy}\n Precision: {precision}\n Recall: {recall}\n F1-Score: {f1}\n Confusion Matrix:\n {cmatrix}\n')

    # Train meta-classifier
    meta_classifier = LogisticRegression()
    meta_classifier.fit(train_meta, y_train)

    # Use predictions as features for meta-classifier
    final_predictions = meta_classifier.predict(test_meta)
    accuracy = accuracy_score(y_test, final_predictions)
    precision = precision_score(y_test, final_predictions, average='macro', zero_division=1)
    recall = recall_score(y_test, final_predictions, average='macro', zero_division=1)
    f1 = f1_score(y_test, final_predictions, average='macro', zero_division=1)
    report = classification_report(y_test, final_predictions)
    print(f'Stacking Classifier:\n {report}\n Accuracy: {accuracy}\n Precision: {precision}\n Recall: {recall}\n F1-Score: {f1}\n Confusion Matrix:\n {cmatrix}\n')

    # Define a boosting classifier with the current classifier as the base estimator
    boosting_clf = AdaBoostClassifier(base_estimator=meta_classifier, n_estimators=50, random_state=42, algorithm='SAMME')

    # Train the boosting classifier on the preprocessed training data
    boosting_clf.fit(train_meta, y_train)

    # Evaluate the performance of the classifier on the preprocessed testing data
    y_pred = boosting_clf.predict(test_meta)

    # Print the classification report and accuracy score
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Precision: {precision_score(y_test, y_pred, average='macro')}")
    print(f"Recall: {recall_score(y_test, y_pred, average='macro')}")
    print(f"F1: {f1_score(y_test, y_pred, average='macro')}")

    '''
    # This code can be used to plot the confusion matrix for any of the models
    cm = confusion_matrix(y_test, y_pred)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, cmap="Blues", fmt='g', cbar=False)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')

    ax.xaxis.set_ticklabels(['Non-Ironic', 'Ironic'])
    ax.yaxis.set_ticklabels(['Non-Ironic', 'Ironic'])
    plt.show()
    '''





