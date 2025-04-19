import pandas as pd
import nltk
from nltk.corpus import wordnet
import random

# Download necessary NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')

# Read the dataset
df = pd.read_csv('Reviews.tsv', sep='\t')

# Define synonym replacement function
def synonym_replacement(sentence):
    words = sentence.split()
    augmented_sentence = []

    for word in words:
        synonyms = wordnet.synsets(word)
        if synonyms:  # If synonyms exist
            synonym = random.choice(synonyms).lemmas()[0].name()  # Get a random synonym
            if synonym != word:  # Avoid replacing with the same word
                augmented_sentence.append(synonym)
            else:
                augmented_sentence.append(word)
        else:
            augmented_sentence.append(word)
    
    return ' '.join(augmented_sentence)

# Check the initial number of records
original_data_size = len(df)
print(f"Original data size: {original_data_size} records.")

# Create a list to store augmented reviews and their corresponding sentiment
augmented_reviews = []
augmented_labels = []

# Number of augmented reviews to generate
num_augmented_reviews = 9000  # We need to augment to get 9000 new reviews

# Loop to generate the augmented reviews
for idx, review in df.iterrows():
    augmented_reviews.append(review['Review'])
    augmented_labels.append(review['Liked'])

# Generate 9000 augmented reviews
while len(augmented_reviews) < original_data_size + num_augmented_reviews:
    for idx, review in df.iterrows():
        augmented_reviews.append(synonym_replacement(review['Review']))
        augmented_labels.append(review['Liked'])
        if len(augmented_reviews) >= original_data_size + num_augmented_reviews:
            break

# Create a new DataFrame with the augmented data
augmented_df = pd.DataFrame({
    'Review': augmented_reviews,
    'Liked': augmented_labels
})

# Save the augmented dataset to a new .tsv file
augmented_df.to_csv('Augmented_Reviews.tsv', sep='\t', index=False)

# Verify the augmented dataset size
print(f"Augmented data size: {len(augmented_df)} records.")
