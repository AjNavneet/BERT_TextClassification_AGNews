import ktrain
from ktrain import text

# Function to perform feature engineering for BERT-based text classification
def perform_feature_engineering(ag_news_train_df, ag_news_test_df, max_length):
    # Use ktrain's 'texts_from_df' function to prepare the text data for BERT model
    (X_train, y_train), (X_test, y_test), preprocessing_var = text.texts_from_df(
        train_df=ag_news_train_df,  # Training DataFrame
        text_column='text',         # Column containing text data
        label_columns='label',      # Column containing labels
        val_df=ag_news_test_df,      # Validation DataFrame
        maxlen=max_length,          # Maximum sequence length
        preprocess_mode='bert'      # Preprocessing mode for BERT
    )

    return (X_train, y_train), (X_test, y_test), preprocessing_var
