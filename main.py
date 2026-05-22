import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix



# DATASET

data = {

    "email": [

        "Congratulations you won free money",

        "Verify your bank account immediately",

        "Limited time offer click here",

        "Urgent login required",

        "Meeting scheduled for tomorrow",

        "Project report attached",

        "Let's have lunch tomorrow",

        "Your Amazon package has arrived",

        "Win cash prize now",

        "Reset your password immediately"

    ],

    "label": [

        "phishing",

        "phishing",

        "phishing",

        "phishing",

        "safe",

        "safe",

        "safe",

        "safe",

        "phishing",

        "phishing"

    ]
}



# CONVERT TO DATAFRAME

df = pd.DataFrame(data)



# FEATURES AND LABELS

x = df["email"]

y = df["label"]



# TEXT CONVERSION

vectorizer = CountVectorizer()

x_vectorized = vectorizer.fit_transform(x)



# SPLIT DATA

x_train, x_test, y_train, y_test = train_test_split(

    x_vectorized,

    y,

    test_size=0.2,

    random_state=42

)



# MODEL

model = MultinomialNB()

model.fit(x_train, y_train)



# PREDICTIONS

predictions = model.predict(x_test)



# ACCURACY

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy * 100, "%")



# CONFUSION MATRIX

cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:")

print(cm)



# EMAIL ANALYZER

while True:

    print("\nPaste the email content below.")

    print("Type END on a new line when finished.\n")

    lines = []

    while True:

        line = input()

        if line == "END":

            break

        lines.append(line)

    user_email = " ".join(lines)



    # TRANSFORM EMAIL

    email_vector = vectorizer.transform([user_email])



    # PREDICT

    result = model.predict(email_vector)

    print("\nPrediction:", result[0])



    # CHECK AGAIN

    again = input("\nCheck another email? (yes/no): ")

    if again.lower() != "yes":

        break