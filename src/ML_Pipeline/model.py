import ktrain
import timeit
from ktrain import text

# Function to create and train a BERT-based text classification model
def create_and_train_bert_model(X_train, y_train, X_test, y_test, preprocessing_var):
    # Create a BERT text classifier model
    transformer_bert_model = text.text_classifier(name='bert',
                                                  train_data=(X_train, y_train),
                                                  preproc=preprocessing_var)
    print("Transformer Layers: \n", transformer_bert_model.layers)
    print("\nCompiling & Training BERT for maxlen=512 & batch_size=6")
    bert_learner = ktrain.get_learner(model=transformer_bert_model,
                                      train_data=(X_train, y_train),
                                      val_data=(X_test, y_test),
                                      batch_size=6)

    start_time = timeit.default_timer()
    print("\nFine Tuning BERT on AG News Dataset with learning rate=2e-5 and epochs=3")
    bert_learner.fit_onecycle(lr=2e-5, epochs=3)
    stop_time = timeit.default_timer()
    print("Total training time in minutes: \n", (stop_time - start_time) / 60)
    return bert_learner

# Function to evaluate the performance of the BERT model
def check_model_performance(bert_learner, class_label_names):
    print("BERT Performance Metrics on AG News Dataset :\n", bert_learner.validate())
    print("BERT Performance Metrics on AG News Dataset with Class Names :\n",
          bert_learner.validate(class_names=class_label_names))
    return None

# Function to save the fine-tuned BERT model
def save_fine_tuned_bert_model(bert_learner, preprocessing_var):
    bert_predictor = ktrain.get_predictor(bert_learner.model, preproc=preprocessing_var)
    bert_predictor.save('../output/bert-ag-news-predictor')
    return None

# Function to reload the saved BERT model
def load_model():
    bert_predictor = ktrain.load_predictor('../output/bert-ag-news-predictor')
    print("Bert model loaded successfully: \n", bert_predictor.get_classes())
    return bert_predictor
