import pandas as pd
import re
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab')

# Download NLTK data needed for tokenization
nltk.download('punkt')

# Load the data from CSV
file_path = 'C:/Users/user/PycharmProjects/text-classification-model/datasets/NewsData.io_Sample_data.csv'
  # Update with the path to your file
data = pd.read_csv(file_path)

# 1. Handling missing values
data.fillna({'keywords': 'Unknown', 'creator': 'Unknown', 'video_url': 'Unknown', 'ai_org': 'Unknown'}, inplace=True)

# 2. Text Cleaning: removing extra spaces and special characters from 'description' and 'content'
def clean_text(text):
    text = re.sub(r'\s+', ' ', str(text)).strip()
    return text

data['description'] = data['description'].apply(clean_text)
data['content'] = data['content'].apply(clean_text)

# 3. Date Formatting for 'pubDate'
data['pubDate'] = pd.to_datetime(data['pubDate'], errors='coerce')

# 4. Tokenizing 'title', 'description', and 'content'
data['title_tokens'] = data['title'].apply(word_tokenize)
data['description_tokens'] = data['description'].apply(word_tokenize)
data['content_tokens'] = data['content'].apply(word_tokenize)

# 5. Saving the preprocessed data to a new CSV
output_file = 'preprocessed_news_data.csv'
data.to_csv(output_file, index=False)

print(f"Preprocessed file saved as {output_file}")
