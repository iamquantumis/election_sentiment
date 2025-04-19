# Twitter Sentiment Analysis for Election Candidates
Currently deployed on Streamlit Community Cloud: [electionsentiment.streamlit.app](https://electionsentiment.streamlit.app/)

Takes in two CSV files with Twitter data, one for each candidate, and performs a ML multi-model ensemble sentiment analysis on the tweets. After anlaysis, it will display a chart comparing counts of both positive and negative sentiments for each candidate and then displays which candidate is more likely to win the election based on most positive votes.

You can choose to use the sample Twitter data files already included or provide your own CSV files, as long as they have a `tweet` column. A small subset of the [US Election 2020 Sample Dataset](https://www.kaggle.com/datasets/manchunhui/us-election-2020-tweets/data) from Kaggle is what is included if you don't already have your own Twitter data files to upload.

The sentiment models it uses are Hugging Face [RoBERTa](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest), [DistilBERT](https://huggingface.co/docs/transformers/en/model_doc/distilbert), and [SieBERT](https://huggingface.co/siebert/sentiment-roberta-large-english).

## Deployment on Streamlit
First clone this repository to your own GitHub account. Then log into your Streamlit Community account (or create one first). [Create a new app](https://share.streamlit.io/new) and connect it to your GitHub repo with this code. You will also need a [Hugging Face API token](https://huggingface.co/settings/tokens) which you should add to your [Streamlit secrets file](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management]) from your App dashboard.
