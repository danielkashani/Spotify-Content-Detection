# Explicit Content Detection on Spotify

Binary classification project that detects explicit songs using lyrics and metadata for over 438,000 Spotify tracks.

## Overview
The objective is to classify whether a song is marked as "explicit" using a combination of text preprocessing and audio-related metadata. This model helps maintain content safety in music applications.

## Features
- Lyrics preprocessing and text cleaning
- Feature engineering from both text and audio tags (e.g., loudness, tempo)
- Trained and evaluated multiple classifiers (e.g., Logistic Regression, Random Forest)
- Iterative data cleaning to improve model performance

## Results
- Best model achieved:
  - **Recall**: 97%
  - Prioritized recall to minimize false negatives in content moderation

## Tools & Technologies
- Python
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn
