# Ensemble Sentiment Analysis for 2020 US Election Tweets
# Updated to use st.session_state to preserve state across reruns in Streamlit
# TRANSFORMERS_NO_TF disables TensorFlow backend to avoid compatibility issues
# with Keras 3 when using models like DistilBERT that may trigger TF imports

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"  # MUST be set before importing transformers

import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Pipeline model packages
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Patch torch bug that sometimes affects Streamlit
torch.classes.__path__ = []

# Used to run on Streamlit
import streamlit as st

# ---------------------------
# Clean tweet text using basic regex and ASCII filtering
# ---------------------------
def clean_tweet(tweet):
    tweet = re.sub(r"http\S+", "", tweet)
    tweet = re.sub(r"@\w+", "", tweet)
    tweet = re.sub(r"#", "", tweet)
    tweet = re.sub(r"\d+", "", tweet)
    tweet = tweet.encode("ascii", "ignore").decode("ascii")
    tweet = tweet.strip()
    return tweet


# ---------------------------
# Load two sentiment analysis models: RoBERTa and DistilBERT
# ---------------------------
@st.cache_resource
def load_sentiment_pipelines():
    try:
        from transformers import pipeline

        roberta_model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        distilbert_model = "distilbert-base-uncased-finetuned-sst-2-english"

        roberta_pipe = pipeline(
            "sentiment-analysis", model=roberta_model, tokenizer=roberta_model, device=-1
        )
        distilbert_pipe = pipeline(
            "sentiment-analysis", model=distilbert_model, tokenizer=distilbert_model, device=-1
        )

        print("✅ Loaded RoBERTa and DistilBERT sentiment models.")
        return roberta_pipe, distilbert_pipe

    except Exception as e:
        print(f"❌ Failed to load sentiment models: {e}")
        return None, None


# Load both pipelines once (cached)
roberta_pipe, distilbert_pipe = load_sentiment_pipelines()

# Global counter for tracking batch number
batch_counter = {"i": 0}

# ---------------------------
# Perform batch sentiment analysis using both models and ensemble logic
# ---------------------------
def analyze_rd(batch, roberta_pipe, distilbert_pipe):
    print(f"Processing batch #{batch_counter['i']}")
    batch_counter["i"] += 1

    # Run inference for each model
    results_roberta = roberta_pipe(batch["cleaned_tweets"])
    results_distilbert = distilbert_pipe(batch["cleaned_tweets"])

    ensemble_sentiments, ensemble_scores, ensemble_votes = [], [], []

    for i in range(len(batch["cleaned_tweets"])):
        label_roberta = results_roberta[i]["label"].upper()
        label_distilbert = results_distilbert[i]["label"].upper()

        # Filter out tweets where RoBERTa is NEUTRAL
        if label_roberta == "NEUTRAL":
            ensemble_sentiments.append(None)
            ensemble_scores.append(None)
            ensemble_votes.append(None)
            continue

        # Majority vote (favor RoBERTa in tie)
        predictions = [label_roberta, label_distilbert]
        vote = max(set(predictions), key=predictions.count)

        score_roberta = results_roberta[i]["score"]
        score_distilbert = results_distilbert[i]["score"]

        maj_scores = [
            score
            for score, label in zip([score_roberta, score_distilbert], predictions)
            if label == vote
        ]

        ensemble_sentiments.append(vote)
        ensemble_scores.append(sum(maj_scores) / len(maj_scores))
        ensemble_votes.append(
            ("R" if label_roberta == vote else "")
            + ("D" if label_distilbert == vote else "")
        )

    return {
        "ensemble_sentiment": ensemble_sentiments,
        "ensemble_score": ensemble_scores,
        "ensemble_votes": ensemble_votes,
    }


# ---------------------------
# Main Streamlit App
# ---------------------------
def main():
    st.title("Ensemble Sentiment Analysis for Two Election Candidates")
    st.write(
        """
        Use the sample data or upload two CSV files, one for each candidate and   
        provide the candidate names. The app will merge the data keeping track of    
        which tweets, apply data cleaning on the tweets, and run an 
        ensemble sentiment analysis using three pre-trained Hugging Face models.
        **Note:** Each CSV file must include a column named `tweet`.
        """
    )

    # Save session state of DFs so they don't reload after every user interaction
    for key in ["merged_df", "user_USAonly", 
                "tweetUSA_dataset", "sentiment_results"]:
        if key not in st.session_state:
            st.session_state[key] = None

    st.write("Choose to use the sample data or upload your own...")

    data_source = st.radio(
        "Choose data source:", ["Use included sample data", 
                                "Upload your own CSV files"]
    )

    # Use included data samples of Trump and Biden Tweets
    if data_source == "Use included sample data":

        candidate1_name = "biden"
        candidate2_name = "trump"

        if st.button("Load Sample Data"):

            with st.spinner("Loading sample Tweets about Joe Biden and Donald Trump..."):
                
                try:
                    df1 = pd.read_csv("input/hashtag_bidensamp.csv", 
                                      lineterminator="\n")
                    df2 = pd.read_csv("input/hashtag_trumpsamp.csv", 
                                      lineterminator="\n")

                    df1["candidate"] = candidate1_name
                    df2["candidate"] = candidate2_name

                    st.session_state.merged_df = pd.concat([df1, df2], 
                                                           ignore_index=True)
                    st.success("Sample data loaded.")

                except Exception as e:
                    st.error(f"Error loading sample data: {e}")

    # Otherwise use user-provided CSV Twitter datasets
    else:
        col1, col2 = st.columns(2)

        with col1:
            candidate1_file = st.file_uploader("Upload CSV for Candidate 1", 
                                               type="csv")
            candidate1_name = st.text_input("Candidate 1 Name")

        with col2:
            candidate2_file = st.file_uploader("Upload CSV for Candidate 2", 
                                               type="csv")
            candidate2_name = st.text_input("Candidate 2 Name")

        if candidate1_file and candidate2_file and\
              candidate1_name and candidate2_name:
            try:
                df1 = pd.read_csv(candidate1_file, index_col=0)
                df2 = pd.read_csv(candidate2_file, index_col=0)

                df1["candidate"] = candidate1_name
                df2["candidate"] = candidate2_name

                st.session_state.merged_df = pd.concat([df1, df2], 
                                                       ignore_index=True)
                st.success("User data uploaded.")

            except Exception as e:
                st.error(f"Error reading uploaded CSVs: {e}")

        else:
            st.warning("Please upload CSV files and provide names for both candidates.")
            return

    if st.session_state.merged_df is not None:
        df = st.session_state.merged_df.copy()
        
        # Shorten any United States (/of America) to simply "US"
        # Check if "country" column exists
        if "country" not in df.columns:
            df['country'] = "US"

        df['country'] = df['country'].replace(
            {'United States of America': "US", 'United States': "US"}
        )

        tweets_cntryUSA = df[df["country"] == "US"]

        # Check to see where user_location is available, but no country specified
        if "user_location" not in df.columns:
            df['user_location'] = "USA"

        tweets_loconly = df[df['country'].isnull() & 
                            df['user_location'].notnull()]

        statelist = [
            "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA",
            "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA",
            "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY",
            "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX",
            "UT", "VT", "VA", "WA", "WV", "WI", "WY", "USA"
        ]
        
        # Filter the rows that have country as null but location as filled for those
        # that have the last two characters matching a State abbreviation
        user_states = tweets_loconly[tweets_loconly['user_location']\
                                        .str[-2:].isin(statelist)]
        
        # Check if user_location indicates "USA" if no state abbreviation at the end
        user_stateUSA = tweets_loconly[tweets_loconly['user_location']\
                                        .str.contains("USA", na=False)]

        # Combine DFs with "US" country, and those with no country but US locations.
        user_USAonly = pd.concat([tweets_cntryUSA, 
                                    user_states, 
                                    user_stateUSA], ignore_index=True)

        # Make sure to fill null 'country' fields with "US"
        user_USAonly['country'] = user_USAonly['country'].fillna(value="US")

        # Create cleaned tweets column
        user_USAonly['cleaned_tweets'] = user_USAonly['tweet'].apply(clean_tweet)

        st.session_state.user_USAonly = user_USAonly

        st.write("### Sampled Merged Data Preview")
        st.dataframe(user_USAonly[['cleaned_tweets', 'candidate']].sample(10))

        # Have option to take only sample of data (runs faster)
        samplesize = st.number_input(
            "Data Sample Size Percent (100 = full dataset)",
            min_value=1,
            max_value=100,
            value=1,
            step=1,
            key="samplesize"
        )

        if 1 <= samplesize <= 100:

            sampled_df = user_USAonly.sample(
                frac=(samplesize / 100), random_state=42
            )

            st.session_state.tweetUSA_dataset = Dataset.from_pandas(sampled_df)

            if st.button("Run Sentiment Analysis"):
                with st.spinner("Running sentiment analysis..."):
                    result_dataset = st.session_state.tweetUSA_dataset.map(
                        lambda batch: analyze_rd(batch, roberta_pipe, distilbert_pipe),
                        batched=True,
                        batch_size=2,
                    )

                    # Convert back to pandas DataFrame for data analysis
                    # Remove any rows that were previously judged as neutral
                    df_results = result_dataset.to_pandas().dropna(
                        subset=["ensemble_score"]
                    )

                    st.session_state.sentiment_results = df_results
                    st.success("Analysis complete!")
        else:
            st.error("Value must be between 1 and 100 inclusive.")
            return

        if st.session_state.sentiment_results is not None:
            
            df_results = st.session_state.sentiment_results
            
            # Provide download option for the results
            st.download_button(
                "Download Results as CSV",
                df_results.to_csv(index=False),
                "sentiment_analysis_results.csv",
            )

            st.write(
                """
                ### Chart of Sentiment Count by Candidate
                Every sentiment judged by the model has a sentiment score which is 
                a percentage of "confidence" the model has about its judgment. We can
                chart the sentiment counts using a minimum confidence score (usually 50%).
                # """)

            # Form to hold slider + submit button
            with st.form("confidence_form"):
                min_conf_percent = st.slider(
                    "Minimum Confidence Percent",
                    min_value=50,
                    max_value=100,
                    value=50,
                    step=1,
                )
                update = st.form_submit_button("Update Chart")

            if update:
                confidence = min_conf_percent / 100

                # Re‑compute counts with chosen confidence
                sentiment_counts = (
                    df_results[df_results["ensemble_score"] > confidence]
                    .groupby("candidate")["ensemble_sentiment"]
                    .value_counts()
                    .unstack(fill_value=0)
                )

                # Create the plot
                fig = plt.figure(figsize=(12, 6))

                # Get the candidates and sentiments
                candidates = sentiment_counts.index
                sentiments = sentiment_counts.columns
                bar_width = 0.25

                # Set the positions of the bars
                x = np.arange(len(candidates))

                # Plot bars for each sentiment
                for i, sentiment in enumerate(sentiments):
                    plt.bar(
                        x + i * bar_width,
                        sentiment_counts[sentiment],
                        bar_width,
                        label=sentiment,
                    )
                
                # Customize the plot
                plt.xlabel("Candidates")
                plt.ylabel("Number of Tweets")
                plt.title(
                    f"Sentiment Count per Candidate (Confidence > {confidence * 100}%)"
                )
                plt.xticks(
                    x + bar_width * (len(sentiments) - 1) / 2,
                    candidates,
                    rotation=45,
                )
                plt.legend(title="Sentiment")
                plt.grid(True, alpha=0.3)

                # Adjust layout to prevent label cutoff
                plt.tight_layout()
                
                # Display the plot in Streamlit
                st.pyplot(fig)


if __name__ == "__main__":
    main()