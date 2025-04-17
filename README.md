### Twitter Sentiment Analysis for Election Candidates
To be deployed on Streamlit Community: [electionsentiment.streamlit.app](https://electionsentiment.streamlit.app/)

Takes in two CSV files with Twitter data, one for each candidate, and performs a ML multi-model ensemble sentiment analysis on the tweets. After anlaysis, it wil display chart comparing both positive and negative sentiments for each candidate and you might able to infer which candidate is more likely to win the election. 

You can choose to use the sample Twitter data files already included or provide your own CSV files, as long as they have a `tweet` column. A small subset of the [US Election 2020 Sample Dataset](https://www.kaggle.com/datasets/manchunhui/us-election-2020-tweets/data) from Kaggle is what is included if you don't already have your own Twitter data files to upload.

The sentiment models it uses are Hugging Face [RoBERTa](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest), [DistilBERT](https://huggingface.co/docs/transformers/en/model_doc/distilbert), and [SieBERT](https://huggingface.co/siebert/sentiment-roberta-large-english).
