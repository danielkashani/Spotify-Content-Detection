# Spotify Explicit Content Detection

Binary classification project to detect explicit songs using lyrics and audio metadata from 438,000+ Spotify tracks.

## Overview
This project trains machine learning models to classify whether a song is marked as explicit, using both text (lyrics) and audio features. The approach focuses on maximizing recall for safe content moderation.

## Dataset

- Spotify tracks dataset (CSV, see [Project.md](./Project.md) for download links)
- Preprocessed lyrics and audio features

## Installation

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
```

## Usage

- Run `content-detection-code.py` after updating the file_id and path to your dataset.
- Example:
  ```python
  import pandas as pd
  df = pd.read_csv('spotify_data.csv')
  # See content-detection-code.py for full modeling pipeline
  ```

## Features

- Lyrics and metadata preprocessing
- Feature engineering (text + audio)
- Model training: Logistic Regression, Random Forest, KNN, ensemble methods
- Prioritization of recall (minimizing false negatives)

## Results

- Best model recall: 97%
- Ensemble approaches (bagging, random forest) yield best results for recall

## Project Structure

- `content-detection-code.py` — Full pipeline: EDA, feature engineering, modeling
- `README.md` — Project summary and quickstart
- `Project.md` — Expanded technical report and code

## License

MIT
