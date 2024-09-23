import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load the dataset
df = pd.read_csv('../datasets/preprocessed_news_data.csv')

# 1. Basic Info and Data Structure
print("Dataset Information:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

print("\nMissing Values Count:")
print(df.isnull().sum())

# 2. Handling Missing Values (choose option based on your data)
# Option 1: Drop missing values
# df_cleaned = df.dropna()

# Option 2: Fill missing values with mean (for numerical data)
df.fillna(df.mean(), inplace=True)

# 3. Analyze Target Variable
# Replace 'target_column' with the actual target column name
print("\nTarget Variable Distribution:")
print(df['target_column'].value_counts())

# Plot Target Variable Distribution
sns.histplot(df['target_column'])
plt.title('Target Variable Distribution')
plt.show()

# 4. Analyze Features
# Plot histograms of all numerical features
df.hist(bins=30, figsize=(15, 10))
plt.suptitle('Numerical Feature Distribution')
plt.show()

# Correlation Matrix (for numerical features)
corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# 5. Text Data Analysis (if applicable)
# Replace 'text_column' with the actual text column name
df['text_length'] = df['text_column'].apply(len)

# Plot Text Length Distribution
sns.histplot(df['text_length'], bins=30)
plt.title('Text Length Distribution')
plt.show()

# Generate Word Cloud (if applicable for textual data)
wordcloud = WordCloud().generate(' '.join(df['text_column']))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Word Cloud')
plt.show()

# 6. Visualize relationship between text length and category (if applicable)
# Replace 'category_column' with the actual category column name
sns.boxplot(x='category_column', y='text_length', data=df)
plt.title('Text Length vs Category')
plt.show()

# 7. Feature Engineering for NLP (if applicable)
df['word_count'] = df['text_column'].apply(lambda x: len(x.split()))
df['unique_word_count'] = df['text_column'].apply(lambda x: len(set(x.split())))

# Summary of new features
print("\nNew Features (Word Count, Unique Word Count):")
print(df[['word_count', 'unique_word_count']].head())

# Save the cleaned data (optional)
df.to_csv('cleaned_data.csv', index=False)

print("EDA Completed!")
