"""Scenario 17: NLP classification of POST-OUTCOME failure descriptions.

IMPORTANT: This is a diagnostic operations use case, NOT an input to the main
pre-settlement failure predictor. failure_reason_description is known after the
outcome and would cause leakage if fed into the predictive classifier.

Task: classify a textual failure description into a normalized reason category.
Methods: TF-IDF + Logistic Regression.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from common import load_data, OUTPUT_DIR
import pandas as pd


def main():
    df=load_data().query("settlement_failed == 1").copy()
    X_train,X_test,y_train,y_test=train_test_split(
        df["failure_reason_description"],df["failure_reason"],test_size=.25,
        random_state=42,stratify=df["failure_reason"])
    pipe=Pipeline([
        ("tfidf",TfidfVectorizer(ngram_range=(1,2),min_df=2,stop_words="english")),
        ("model",LogisticRegression(max_iter=1000,class_weight="balanced"))])
    pipe.fit(X_train,y_train)
    pred=pipe.predict(X_test)
    report=classification_report(y_test,pred,output_dict=True,zero_division=0)
    pd.DataFrame(report).T.to_csv(OUTPUT_DIR/"17_nlp_classification_report.csv")
    print(classification_report(y_test,pred,zero_division=0))

if __name__ == "__main__":
    main()
