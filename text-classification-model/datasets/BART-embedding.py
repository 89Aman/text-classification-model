import pandas as pd
import torch
from transformers import BertTokenizer, BertModel

# Load the pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Load your CSV file (replace with your actual path)
file_path = 'preprocessed_news_data.csv'
data = pd.read_csv(file_path)


# Function to get BERT embeddings for a text
def get_bert_embedding(text):
    # Tokenize and encode the input text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)

    # Get the hidden states from BERT
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the last hidden state (embedding) of the `[CLS]` token (first token in the sequence)
    cls_embedding = outputs.last_hidden_state[:, 0, :]

    return cls_embedding.squeeze().numpy()


# Apply the embedding function to the 'title', 'description', or 'content' column
# Choose one of the columns for embedding. Here we use 'content' as an example.
data['content_embedding'] = data['content'].apply(lambda x: get_bert_embedding(str(x)))

# Save the dataframe with embeddings to a new file
output_file = 'bert_embeddings_news_data.csv'
data.to_csv(output_file, index=False)

print(f"BERT embeddings saved to {output_file}")
