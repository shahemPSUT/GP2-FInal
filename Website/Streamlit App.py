import pickle

import numpy as np
import pandas as pd
import streamlit as st
from nltk import word_tokenize, PorterStemmer
from nltk.corpus import stopwords


@st.cache_data
def load_data():
    print("Loading the Dataset...")
    df = pd.read_csv("Complete.csv", low_memory=False)
    print("Loading the Model...")
    dtc = pickle.load(open("Decision Tree Model.pickle", "rb"))
    print("Loading the Label Encoder")
    le = pickle.load(open("le.pickle", "rb"))
    print("Loading the Vectorizer")
    vectorizer = pickle.load(open("vectorizer.pickle", "rb"))
    return df, dtc, le, vectorizer


# Define the preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # Perform stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    # Join the tokens back into a single string
    preprocessed_text = ' '.join(stemmed_tokens)
    return preprocessed_text


# Define your predict function here
def predict(username="", user_id="", df=None, dtc=None, le=None, vectorizer=None):
    print("username", username)
    print("user_id", user_id)
    if username != "":
        user_data = df[df['screen_name'] == username]
    else:
        user_data = df[df['user_id'] == int(user_id)]

    if user_data.shape[0] == 0:
        return None

    user_data.dropna(inplace=True)

    print("Reading the user data.....")
    user_text = user_data.text.apply(preprocess_text)

    print("Examining tweets")
    vectorized = vectorizer.transform(user_text)
    vectorized = np.array(vectorized.toarray())
    vectorized_df = pd.DataFrame(vectorized, columns=vectorizer.get_feature_names_out())
    user_data = user_data.reset_index()
    com_df = pd.concat([vectorized_df, user_data], axis=1)

    print("Preprocessing the textual columns")
    com_df.drop(["text", "index"], axis=1, inplace=True)
    str_cols = [col for col in com_df.columns if com_df[col].dtype == object]

    for col in str_cols:
        # print("col:", col)
        com_df[col] = le[col].transform(com_df[col])

    X = com_df.iloc[:, :-1]
    # y = com_df.iloc[:, -1]

    # print(X.shape)

    print("Making the prediction")
    prediction = dtc.predict(X)
    counts = pd.Series(prediction).value_counts()

    counts = pd.Series(prediction).value_counts()

    if len(counts.index) == 1:
        print("Account is ", le['Label'].classes_[counts.index[0]])
        return le['Label'].classes_[counts.index[0]]
    # elif counts[1] > counts[0]:
    #     print("Account is Real")
    # else:
    #     print("Account is Fake")
    elif counts[1] > counts[0]:
        print("This account is real")
        return "Real"
    else:
        print("This account is Fake")
        return "Fake"


# Streamlit app code
def main():
    st.title("TrueTwitter")
    st.header("Detect Fake Twitter Accounts")

    st.write("Welcome to our website, where you can detect fake Twitter accounts! Our powerful algorithm analyzes"
             "various factors and patterns to determine the authenticity of a Twitter account. Enter a Twitter username"
             " below and let us do the work for you.")

    st.subheader("How it Works")
    st.markdown("1. Enter the Twitter username or user_id of the account you want to check.\n"
                "2. Our system will gather relevant information and examine multiple indicators.\n"
                "3. Based on our analysis, we will provide you with a classification of the account's authenticity.")

    st.subheader("Why Detect Fake Accounts?")
    st.markdown(
        "With the increasing prevalence of fake accounts on social media platforms, it's crucial to have tools that"
        " can identify them. Fake accounts can spread misinformation, engage in fraudulent activities, or manipulate"
        " public opinion. By detecting and exposing these accounts, we contribute to maintaining a trustworthy and"
        " reliable online environment.")

    st.subheader("Benefits of Using Our Service")
    st.markdown(
        "- Save time and effort by relying on our sophisticated algorithm to determine the authenticity of Twitter accounts.\n"
        "- Make informed decisions when engaging with users on Twitter, especially for businesses, influencers, and"
        " individuals seeking credible connections.\n"
        "- Contribute to a safer online community by raising awareness about fake accounts and promoting transparency.")

    st.title("Account Verifier")
    st.write("Enter a username to check if it's real or fake.")

    df, dtc, le, vectorizer = load_data()

    # User input
    username = st.text_input("Username")
    st.text("OR")
    user_id = st.text_input("user_id")

    # Use the button
    if st.button("Verify"):

        # Perform prediction
        if username != "":
            result = predict(username,"", df, dtc, le, vectorizer)
            if result is not None:
                st.write(f"The account '{username}' is {result}.")
            else:
                st.write("User not exists!")
        elif user_id != "":
            result = predict("", user_id, df, dtc, le, vectorizer)
            if result is not None:
                st.write(f"The account '{user_id}' is {result}.")
            else:
                st.write("User not exists!")
        else:
            st.write("Please input username or user_id")


if __name__ == "__main__":
    main()
