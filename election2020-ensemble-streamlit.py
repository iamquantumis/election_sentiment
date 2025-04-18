# Ensemble Sentiment Analysis for 2020 US Election Tweets
# Updated to use st.session_state to preserve state across reruns in Streamlit
# TRANSFORMERS_NO_TF disables TensorFlow backend to avoid compatibility issues
# with Keras 3 when using models like DistilBERT that may trigger TF imports

import os
# os.environ["TRANSFORMERS_NO_TF"] = "1"  # MUST be set before importing transformers

import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Pipeline model packages
# import torch - Don't need if using HF API
from datasets import Dataset
# from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Patch torch bug that sometimes affects Streamlit
# torch.classes.__path__ = []

# Used to run on Streamlit
import streamlit as st

# New import for HF Inference API
from huggingface_hub import InferenceApi

# ---------------------------
# Configuration: set HF API token
# ---------------------------

# Read from Streamlit secrets
hf_api_token = st.secrets.get("HF_API_TOKEN") or os.getenv("HF_API_TOKEN")

if not hf_api_token:
    st.error(
        "Hugging Face API token not found. Please set HF_API_TOKEN "
        "Streamlit secrets or in your environment."
    )
    st.stop()
# ---------------------------

# ---------------------------
# Clean tweet text using basic regex and ASCII filtering
# ---------------------------
def clean_tweet(tweet):

    # Remove URLs
    tweet = re.sub(r"http\S+", "", tweet)
    # Remove mentions
    tweet = re.sub(r"@\w+", "", tweet)
    # Remove hashtags
    tweet = re.sub(r"#", "", tweet)
    # Remove numbers
    tweet = re.sub(r"\d+", "", tweet)
    # Strip non-ASCII characters
    tweet = tweet.encode("ascii", "ignore").decode("ascii")
    # Trim whitespace
    return tweet.strip()

# ---------------------------
# Load sentiment clients via HF Inference API
# ---------------------------
@st.cache_resource
def load_sentiment_pipelines():
    """Initialize HF Inference API clients for sentiment analysis."""
    try:

        roberta_api = InferenceApi(
            repo_id="cardiffnlp/twitter-roberta-base-sentiment-latest",
            token=hf_api_token,
            # task="sentiment-analysis"
        )
        distilbert_api = InferenceApi(
            repo_id="distilbert-base-uncased-finetuned-sst-2-english",
            token=hf_api_token,
            # task="sentiment-analysis"
        )
        siebert_api = InferenceApi(
        repo_id="siebert/sentiment-roberta-large-english",
        token=hf_api_token
        )

        return roberta_api, distilbert_api, siebert_api
    
    except Exception as e:
        st.error(f"Failed to initialize HF Inference API: {e}")
        return None, None, None

# Load both clients once (cached)
roberta_pipe, distilbert_pipe, siebert_pipe = load_sentiment_pipelines()

# Global counter for tracking batch number
batch_counter = {"i": 0}

# ---------------------------
# Perform batch sentiment analysis using both clients and ensemble logic
# ---------------------------
def analyze_rds(batch, roberta_pipe, distilbert_pipe, siebert_pipe):
    """Runs the two inference clients on a batch and applies ensemble voting."""

    print(f"Processing batch #{batch_counter['i']}")
    batch_counter["i"] += 1

    tweets = batch["cleaned_tweets"]

    # Call HF Inference API for the cleaned tweets batch
    # If each item is itself a single‑element list, unwrap it:
    raw_roberta = roberta_pipe(inputs=tweets, params={"top_k": 1})
    raw_distilbert = distilbert_pipe(inputs=tweets, params={"top_k": 1})
    raw_siebert = siebert_pipe(inputs=tweets, params={"top_k": 1})
    
    # Inference API may already flatten to dicts. To be safe:
    results_roberta = [r[0] if isinstance(r, list) 
                       else r for r in raw_roberta]
    
    results_distilbert = [r[0] if isinstance(r, list) 
                          else r for r in raw_distilbert]
    
    results_siebert = [r[0] if isinstance(r, list) 
                       else r for r in raw_siebert]

    ensemble_sentiments, ensemble_scores, ensemble_votes = [], [], []

    for i in range(len(tweets)):
        # Uppercase labels for consistency
        label_roberta = results_roberta[i]["label"].upper()
        label_distilbert = results_distilbert[i]["label"].upper()
        label_siebert = results_siebert[i]["label"].upper()

        # Skip instances where RoBERTa yields NEUTRAL
        if label_roberta == "NEUTRAL":
            ensemble_sentiments.append(None)
            ensemble_scores.append(None)
            ensemble_votes.append(None)
            continue

        # Majority vote min 2-of-3
        preds = [label_roberta, label_distilbert, label_siebert]
        vote = max(set(preds), key=preds.count)

        # Retrieve corresponding confidence scores
        score_roberta = results_roberta[i]["score"]
        score_distilbert = results_distilbert[i]["score"]
        score_siebert = results_siebert[i]["score"]

        scores = [score_roberta, score_distilbert, score_siebert]

        maj_scores = [
            score for score, lbl in zip(scores, preds) if lbl == vote
        ]

        # Append ensemble votes
        ensemble_sentiments.append(vote)
        
        # Avg sentiment score across winning models
        ensemble_scores.append(sum(maj_scores) / len(maj_scores)) 

        # Label which two or three models agree
        ensemble_votes.append(
            ("R" if label_roberta == vote else "") +
            ("D" if label_distilbert == vote else "") +
            ("S" if label_siebert == vote else "")
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
    st.image("support/vote-scrabble.jpg",
             caption="Image licensed with [Unsplash](https://unsplash.com)")
    st.write(
        """
        Can we predict the likely outcome of elections by performing sentiment analysis
        on the tweets about each candidate?
        
        Use the sample data or upload two CSV files—one per candidate. The app merges
        tweets, cleans them, and sends batches to Hugging Face Inference API for
        sentiment analysis. Results are ensembled across the
        [RoBERTa](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest), 
        [DistilBERT](https://huggingface.co/docs/transformers/en/model_doc/distilbert) and 
        [SieBERT](https://huggingface.co/siebert/sentiment-roberta-large-english) inference models.
        
        **Note:** Each CSV must include a `tweet` column.
        """
    )

    # Initialize session-state variables
    for key in ["merged_df", 
                "user_USAonly", 
                "tweetUSA_dataset", 
                "sentiment_results"]:
        
        if key not in st.session_state:
            st.session_state[key] = None

    st.write("#### Step 1: Choose Data Source")
    data_source = st.radio(
        "Choose data source:", ["Use included sample data", 
                                "Upload your own CSV files"]
    )

    # --- Sample data path
    if data_source == "Use included sample data":

        candidate1_name, candidate2_name = "biden", "trump"

        if st.button("Load Sample Data"):
            with st.spinner("Loading sample tweets..."):
                try:
                    df1 = pd.read_csv("input/hashtag_bidensamp.csv", lineterminator="\n")
                    df2 = pd.read_csv("input/hashtag_trumpsamp.csv", lineterminator="\n")

                    df1["candidate"], df2["candidate"] = candidate1_name, candidate2_name

                    st.session_state.merged_df = pd.concat([df1, df2], ignore_index=True)
                    st.success("Sample data loaded.")

                except Exception as e:

                    st.error(f"Error loading sample data: {e}")
    
    # --- Upload path
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            candidate1_file = st.file_uploader("Upload CSV for Candidate 1", type="csv")
            candidate1_name = st.text_input("Candidate 1 Name")

        with col2:
            candidate2_file = st.file_uploader("Upload CSV for Candidate 2", type="csv")
            candidate2_name = st.text_input("Candidate 2 Name")

        if candidate1_file and candidate2_file and candidate1_name and candidate2_name:
            try:
                df1 = pd.read_csv(candidate1_file, index_col=0)
                df2 = pd.read_csv(candidate2_file, index_col=0)

                df1["candidate"], df2["candidate"] = candidate1_name, candidate2_name

                st.session_state.merged_df = pd.concat([df1, df2], ignore_index=True)
                st.success("User data uploaded.")

            except Exception as e:
                st.error(f"Error reading uploaded CSVs: {e}")
        else:
            st.warning("Please upload both CSVs and enter candidate names.")
            return

    # --- Data cleaning & preparation
    if st.session_state.merged_df is not None:
        df = st.session_state.merged_df.copy()

        # Shorten any United States (/of America) to simply "US"
        if "country" not in df.columns:
            df["country"] = "US"
        df["country"] = df["country"].replace(
            {"United States of America": "US", "United States": "US"}
        )

        # Filter US tweets by country or user_location      
        tweets_country = df[df["country"] == "US"]
        if "user_location" not in df.columns:
            df["user_location"] = ""
        tweets_loc = df[df["country"].isnull() & 
                        df["user_location"].notnull()]

        # State abbreviations list
        state_abbrevs = [
            "AL","AK","AZ","AR","CA","CO","CT","DE","DC","FL","GA",
            "HI","ID","IL","IN","IA","KS","KY","LA","ME","MD","MA",
            "MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY",
            "NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX",
            "UT","VT","VA","WA","WV","WI","WY","USA"
        ]

        # Filter the rows that have country as null but location as filled for those
        # that have the last two characters matching a State abbreviation
        # or contain "USA"

        loc_states = tweets_loc[tweets_loc["user_location"]\
                                .str[-2:].isin(state_abbrevs)]
        loc_usa = tweets_loc[tweets_loc["user_location"]\
                             .str.contains("USA", na=False)]

        # Combine all US tweets
        user_USAonly = pd.concat([tweets_country, loc_states, loc_usa], ignore_index=True)
        
        user_USAonly["country"] = user_USAonly["country"].fillna("US")

        # Clean tweets
        user_USAonly["cleaned_tweets"] = user_USAonly["tweet"].apply(clean_tweet)
        st.session_state.user_USAonly = user_USAonly

        # Minimize relevant columsn to keep
        keepcols = ["cleaned_tweets", "candidate"]

        st.write("#### Sampled Data Preview")
        st.dataframe(user_USAonly[keepcols].sample(10))

        # Sampling control
        st.write("#### Step 2: Choose Sample Size")
        sample_pct = st.number_input(
            "Data Sample Size Percent (100 = full dataset)",
            min_value=1, max_value=100, value=1, step=1, key="samplesize"
        )
      
        if not (1 <= sample_pct <= 100):
            st.error("Sample percent must be between 1 and 100.")
            return
          
        sample_df = user_USAonly[keepcols].sample(frac=(sample_pct/100), random_state=42)
 
        st.session_state.tweetUSA_dataset = Dataset.from_pandas(sample_df)

        # Trigger sentiment analysis
        if st.button("Run Sentiment Analysis"):
            with st.spinner("Running sentiment analysis..."):
                
                # Setting "optimal" batch size to reduce cost
                BATCH_SIZE = 200

                result_dataset = \
                    st.session_state.tweetUSA_dataset.map(
                    lambda batch: analyze_rds(batch, 
                                             roberta_pipe, 
                                             distilbert_pipe,
                                             siebert_pipe),
                    batched=True, 
                    batch_size=BATCH_SIZE
                )

                df_results = result_dataset\
                    .to_pandas()\
                    .dropna(subset=["ensemble_score"])
                
                st.session_state.sentiment_results = df_results
                st.success("Analysis complete!")

    # --- Results display & charting
    if st.session_state.sentiment_results is not None:
        
        df_results = st.session_state.sentiment_results

        st.download_button(
            "Download Results as CSV",
            df_results.to_csv(index=False),
            "sentiment_analysis_results.csv"
        )

        st.write(
            """
            #### Chart of Sentiment Count by Candidate
            Each sentiment has an associated confidence score. Use the slider below
            to filter out low-confidence predictions.
            """
        )

        with st.form("confidence_form"):
            min_conf = st.slider(
                "Minimum Confidence Percent", min_value=50, max_value=100, value=50, step=1
            )
            update = st.form_submit_button("Update Chart")

        if update:

            thresh = min_conf / 100

            # Re‑compute counts with chosen confidence
            sentiment_counts = (
                df_results[df_results["ensemble_score"] > thresh]
                .groupby("candidate")["ensemble_sentiment"]
                .value_counts()
                .unstack(fill_value=0)
            )

            # Create the plot
            fig, ax = plt.subplots(figsize=(12, 6))

            # Get the candidates and sentiments
            cands = sentiment_counts.index
            sens = sentiment_counts.columns

            # Set the positions of the bars
            x = np.arange(len(cands))
            width = 0.25

            # Plot bars for each sentiment
            for i, s in enumerate(sens):
                ax.bar(x + i*width, 
                       sentiment_counts[s], 
                       width, 
                       label=s)
                
            # Customize the plot    
            ax.set_xlabel("Candidates")
            ax.set_ylabel("Number of Tweets")
            ax.set_title(f"Sentiment Count per Candidate (Confidence > {min_conf}%)")
            ax.set_xticks(x + width*(len(sens)-1)/2)
            ax.set_xticklabels(cands, rotation=45)
            ax.legend(title="Sentiment")
            ax.grid(True, alpha=0.3)

            # Adjust layout to prevent label cutoff
            plt.tight_layout()

            # Display the plot in Streamlit
            st.pyplot(fig)

            # Get the "Winning" Candidate
            top_positive_candidate = sentiment_counts["POSITIVE"].idxmax()
            top_positive_count     = sentiment_counts["POSITIVE"].max()

            st.write(
                f"####🏆 Candidate with the highest count of POSITIVE tweets is: "
                f"####**{top_positive_candidate}** (with {top_positive_count} positive tweets)"
                f"**NOTE:** These results are based on a small sample of tweets and are for"
                f"educational and entertainment purposes only. There is no guarantee of accuracy."
            )

if __name__ == "__main__":
    main()
