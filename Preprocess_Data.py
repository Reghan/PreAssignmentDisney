import pandas as pd
import numpy as np
import re

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    valid_ratings = data[data['Rating'].between(1, 5)]['Rating']
    median_rating = valid_ratings.median()
    data['Rating'] = data['Rating'].apply(lambda x: median_rating if x > 5 else x)

    data['Year_Month'] = data['Year_Month'].apply(lambda x: x if x.count('-') == 1 else '-'.join(x.split('-')[:2]))

    def standardize_branch(name):
        replacements = {
            'Paris': 'Disneyland_Paris',
            'Hongkong': 'Disneyland_HongKong',
            'California': 'Disneyland_California'
        }
        for key, value in replacements.items():
            if key in name:
                return value
        return name
    data['Branch'] = data['Branch'].apply(standardize_branch)

    def clean_text(text):
        text = str(text)  # Ensure text is treated as a string
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove non-alphanumeric characters
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces and strip trailing and leading spaces
        return text if text else 'No review provided'
    data['Review_Text'] = data['Review_Text'].apply(clean_text)

    data['Rating'] = data['Rating'].apply(lambda x: np.round(x) if not pd.isna(x) else x).astype(pd.Int8Dtype())

    return data
