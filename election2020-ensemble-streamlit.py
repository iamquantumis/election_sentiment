import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Pipeline model packages
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import torch
# Prevent Streamlit’s watcher from iterating torch.classes.__path__
torch.classes.__path__ = []

# Used to run on Streamlit
import streamlit as st

# ---------------------------
# Function to clean tweets
def clean_tweet(tweet):
    tweet = re.sub(r'http\S+', '', tweet)           # Remove URLs
    tweet = re.sub(r'@\w+', '', tweet)               # Remove mentions (@username)
    tweet = re.sub(r'#', '', tweet)                  # Remove the '#' symbol
    tweet = re.sub(r'\d+', '', tweet)                # Remove numbers
    tweet = tweet.encode('ascii', 'ignore').decode('ascii')  # Remove non-ASCII characters
    tweet = tweet.strip()                            # Remove leading/trailing spaces
    return tweet

# ---------------------------
# Cache the loading of sentiment models for efficiency
@st.cache_resource
def load_sentiment_pipelines():
    device = 0 if torch.cuda.is_available() else -1

    # --- Initialize roBERTa model
    sentiment_roberta = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=device,
        truncation=True,  # Automatically truncates to max allowed tokens
        max_length=512    # Defaults to model's max tokens per tweet 
    )

    # --- Initialize DistilBERT model
    distilbert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    if device == 0:
        distilBERT_model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english",
            torch_dtype=torch.float16,
            attn_implementation="sdpa"
        ).to(device)
    else:
        distilBERT_model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        )

    sentiment_distilbert = pipeline(
        "sentiment-analysis",
        model=distilBERT_model,
        tokenizer=distilbert_tokenizer,
        device=device,
        torch_dtype=torch.float16 if device==0 else None
    )

    # --- Initialize SieBERT model
    model_siebert = "siebert/sentiment-roberta-large-english"
    siebert_tokenizer = AutoTokenizer.from_pretrained(model_siebert)
    siebert_model = AutoModelForSequenceClassification.from_pretrained(model_siebert)
    sentiment_siebert = pipeline(
        "sentiment-analysis",
        model=siebert_model,
        tokenizer=siebert_tokenizer,
        device=device
    )

    return sentiment_roberta, sentiment_distilbert, sentiment_siebert


# --- Assign the model variables
sentiment_roberta, sentiment_distilbert, sentiment_siebert = load_sentiment_pipelines()

# ---------------------------
# Ensemble sentiment analysis function

batch_counter = {"i": 0}

def analyze_ensemble(batch):

    # print to console (stdout) for each batch
    print(f"🔎 Processing batch #{batch_counter['i']}")
    batch_counter["i"] += 1

    # Run batch inference for each model
    results_roberta = sentiment_roberta(batch["cleaned_tweets"])
    results_distilbert = sentiment_distilbert(batch["cleaned_tweets"])
    results_siebert = sentiment_siebert(batch["cleaned_tweets"])
    
    ensemble_sentiments = []
    ensemble_scores = []
    ensemble_votes = []
    
    for i in range(len(batch["cleaned_tweets"])):
        label_roberta = results_roberta[i]["label"].upper()
        label_distilbert = results_distilbert[i]["label"].upper()
        label_siebert = results_siebert[i]["label"].upper()

        # Filter out tweets where roBERTa returns NEUTRAL
        if label_roberta == "NEUTRAL":
            ensemble_sentiments.append(None)
            ensemble_scores.append(None)
            ensemble_votes.append(None)
            continue

        # Gather predictions from all three models    
        predictions = [label_roberta, label_distilbert, label_siebert]

        # Majority vote logic: vote is the label that appears most
        vote = max(set(predictions), key=predictions.count)

        # For pure binary predictions (POSITIVE/NEGATIVE) this will 
        # always yield at least a 2-of-3 majority.
        final_sentiment = vote
        
        # Compute ensemble scores: average the scores of the models that agree with the majority vote.
        score_roberta = results_roberta[i]["score"]
        score_distilbert = results_distilbert[i]["score"]
        score_siebert = results_siebert[i]["score"]

        scores = [score_roberta, score_distilbert, score_siebert]

        maj_scores = [score for score, 
                      lab in zip(scores, predictions) if lab == vote]

        ensemble_score = sum(maj_scores) / len(maj_scores)

        # Determine which models contributed to the majority vote by concatenating their initials.
        vote_models = []
        if label_roberta == vote:
            vote_models.append("R")
        if label_distilbert == vote:
            vote_models.append("D")
        if label_siebert == vote:
            vote_models.append("S")
        voting_set = ''.join(vote_models)
        
        # Append the results to our lists
        ensemble_sentiments.append(final_sentiment)
        ensemble_scores.append(ensemble_score)
        ensemble_votes.append(voting_set)
    
    return {
        "ensemble_sentiment": ensemble_sentiments,
        "ensemble_score": ensemble_scores,
        "ensemble_votes": ensemble_votes
    }


# ---------------------------
# Main Streamlit App Function
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
    
    cand1_df = pd.DataFrame()
    cand2_df = pd.DataFrame()
    candidate1_name = 'biden' # If using included sample data
    candidate2_name = 'trump' # If using included sample data

    usesample = st.checkbox("Use included sample data?")

    if st.button("Load Data"):
        if usesample:
            # Reading Trump Dataset 
            cand1_df = pd.read_csv("input/hashtag_bidensamp.csv", lineterminator='\n')

            # Reading Biden Dataset 
            cand2_df = pd.read_csv("input/hashtag_trumpsamp.csv", lineterminator='\n') 
   
        else:
            # Layout for two file uploaders and candidate name inputs
            col1, col2 = st.columns(2)

            with col1:
                candidate1_file = st.file_uploader("Upload CSV for Candidate 1", type="csv", key="candidate1_file")
                candidate1_name = st.text_input("Candidate 1 Name", key="candidate1_name")
            with col2:
                candidate2_file = st.file_uploader("Upload CSV for Candidate 2", type="csv", key="candidate2_file")
                candidate2_name = st.text_input("Candidate 2 Name", key="candidate2_name")

            if candidate1_file and candidate2_file and candidate1_name and candidate2_name:
                try:
                    cand1_df = pd.read_csv(candidate1_file, index_col=0)
                    cand2_df = pd.read_csv(candidate2_file, index_col=0)
                except Exception as e:
                    st.error(f"Error reading one of the CSV files: {e}")
                    return
                
                # Validate that each dataframe has a 'tweet' column
                if "tweet" not in cand1_df.columns or "tweet" not in cand2_df.columns:
                    st.error("Both CSV files must contain a 'tweet' column.")
                    return
            else:
                st.info("Please upload CSV files and provide names for both candidates.")

    if st.button("Clean Data"):
        # Assign candidate names
        cand1_df['candidate'] = candidate1_name
        cand2_df['candidate'] = candidate2_name

        # Merge the two dataframes
        merged_df = pd.concat([cand1_df, cand2_df]).reset_index(drop=True)

        # Shorten any United States (/of America) to simply "US"
        merged_df['country'] = merged_df['country'].replace({'United States of America': "US",'United States': "US"}) 

        # Isolate tweets where `country` is "US"
        tweets_cntryUSA = merged_df[merged_df["country"] == "US"]

        # Check to see where user_location is available, but no country specified
        tweets_loconly = merged_df[merged_df['country'].isnull() & 
                                    merged_df['user_location'].notnull()]
        
        # Provide list of US State abbreviations to parse user_location
        statelist = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA', 
                    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 
                    'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 
                    'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 
                    'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'USA']
        
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
                                    user_stateUSA]).reset_index(drop=True)

        # Make sure to fill null 'country' fields with "US"
        user_USAonly['country'] = user_USAonly['country'].fillna(value="US")

        # Create cleaned tweets column
        user_USAonly['cleaned_tweets'] = user_USAonly['tweet'].apply(clean_tweet)

        # Have option to take only sample of data (runs faster)
        samplesize = st.number_input("Data Sample Size Percent (100 = full dataset)", key="samplesize")
        if samplesize < 1 or samplesize > 100:
            st.error("Value must be between 1 and 100 inclusive.")
            return
        
        # user_USAsample = user_USAonly.sample(frac=(samplesize/100), random_state=42)

        st.write("### Sampled Merged Data Preview")
        st.dataframe(user_USAonly.sample(10))

        # Convert pandas DataFrame into Hugging Face Dataset
        tweetUSA_dataset = Dataset.from_pandas(user_USAonly.sample(frac=(samplesize/100), random_state=42))

    if st.button("Run Sentiment Analysis"):
        with st.spinner("Loading models and running sentiment analysis..."):
            # sentiment_roberta, sentiment_distilbert, sentiment_siebert = load_sentiment_pipelines()
            # result_dataset = analyze_ensemble(tweetUSA_dataset, sentiment_roberta, sentiment_distilbert, sentiment_siebert)
            BATCH_SIZE = 16
            result_dataset_showmodels = tweetUSA_dataset.map(
                            analyze_ensemble,
                            batched=True,
                            batch_size=BATCH_SIZE  # Adjust based on GPU memory/resources
                        )
        st.success("Analysis complete!")
        st.write("### Sentiment Analysis Results")

        # Convert back to pandas DataFrame for data analysis
        tweetUSA_sentiments_showmodels = result_dataset_showmodels.to_pandas()

        # Remove any rows that were previously judged as neutral, now None or NaN
        tweetUSA_sentiments_modelsclean = tweetUSA_sentiments_showmodels\
            .dropna(subset=['ensemble_score'])

        # Provide download option for the results
        csv_result = tweetUSA_sentiments_modelsclean.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Results as CSV",
            data=csv_result,
            file_name='sentiment_analysis_results.csv',
            mime='text/csv'
        )

        if tweetUSA_sentiments_modelsclean.empty:
            st.warning("No tweets to analyze after filtering.")
        else:
            # Let's chart the data by tweet count
            st.write("### Sentiment Count by Candidate")

            confidence = 0.5
            sentiment_counts = (tweetUSA_sentiments_modelsclean[tweetUSA_sentiments_modelsclean['ensemble_score'] > confidence]
                            .groupby('candidate')['ensemble_sentiment']
                            .value_counts()
                            .unstack(fill_value=0))

            # Create the plot
            plt.figure(figsize=(12, 6))

            # Get the candidates and sentiments
            candidates = sentiment_counts.index
            sentiments = sentiment_counts.columns
            n_sentiments = len(sentiments)
            bar_width = 0.25  # Width of each bar

            # Set the positions of the bars
            x = np.arange(len(candidates))

            # Plot bars for each sentiment
            for i, sentiment in enumerate(sentiments):
                plt.bar(x + i * bar_width, 
                        sentiment_counts[sentiment], 
                        bar_width, 
                        label=sentiment)

            # Customize the plot
            plt.xlabel('Candidates')
            plt.ylabel('Number of Tweets')
            plt.title(f'Sentiment Count per Candidate (Confidence > {confidence * 100}%)')
            plt.xticks(x + bar_width * (n_sentiments-1)/2, candidates, rotation=45)
            plt.legend(title='Sentiment')
            plt.grid(True, alpha=0.3)

            # Adjust layout to prevent label cutoff
            plt.tight_layout()

            # Display the plot
            plt.show()

if __name__ == "__main__":
    main()