import pandas as pd
import re

df = pd.read_csv("spam_dataset.csv")
# print(df.head())

def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-z ]', '', text)
    return text

df['text'] = df['text'].apply(clean)
# print(df.head())


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
X = vectorizer.fit_transform(df['text'])
y = df['label']
# print(X)
# print(vectorizer.vocabulary_) 
# print(y.value_counts())
# print(y.shape)


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# print("Train and Test shapes X:")
# print(X_train)
# print("----------------")
# print(X_test) 

# print("Train and Test shapes Y:")
# print(y_train.shape)
# print("----------------")
# print(y_test.shape)

model = MultinomialNB()
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))


new_email = ["Unlock your course materials 50% off"]
new_email_clean = [clean(new_email[0])]

# Convert text → numbers using SAME vectorizer
new_email_vec = vectorizer.transform(new_email)

# Predict
prediction = model.predict(new_email_vec)

print("Email text:", new_email[0])
print("Prediction:", prediction[0])