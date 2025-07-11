# SMS Spam Detection System

A machine learning-powered web application that classifies SMS messages as spam or legitimate with 98.6% accuracy.

## Features

- **Real-time Classification**: Web interface for instant spam detection
- **Advanced ML Pipeline**: XGBoost classifier with engineered features
- **REST API**: Flask-based API for programmatic access
- **High Accuracy**: 98.6% accuracy with 0.97 F1-score
- **Feature Engineering**: 10+ features including URL/email detection, character ratios, and linguistic patterns

## Architecture

The system consists of two main components:
1. **Training Module** (`model_trainer.py`): Trains the XGBoost classifier
2. **Prediction Service** (`app.py` + `predictor.py`): Flask web application for real-time predictions

## Installation

```bash
# Clone the repository
git clone https://github.com/adityakamath255/sms-spam-identifier.git
cd sms-spam-identifier

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('wordnet')"
```

## Usage

### Training a Model

```bash
# Note: You'll need to provide your own spam.csv dataset
# Dataset should have columns: TEXT, LABEL (ham/spam)
python model_trainer.py
```

### Running the Web Application

```bash
python app.py
```

Navigate to `http://localhost:5000` to use the web interface.

## Dataset

The original training dataset is not included in this repository due to inappropriate content. To train your own model:

1. Use a public SMS spam dataset like the [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
2. Ensure your CSV has columns: `TEXT` (message content) and `LABEL` (ham/spam)
3. Place it as `spam.csv` in the project root

## Model Performance

- **Accuracy**: 98.6%
- **F1 Score**: 0.97
- **Training Size**: 5,000+ messages
- **Features**: TF-IDF (1-4 grams) + 10 engineered features

## Technical Details

### Feature Engineering
- URL and email pattern detection
- Character-level ratios (uppercase, digits, special characters)
- Message length and word count statistics
- Phone number detection

### Technologies
- **ML Framework**: XGBoost
- **Web Framework**: Flask
- **Frontend**: HTML, CSS, JavaScript
- **Data Processing**: pandas, scikit-learn, NLTK

## Project Structure

```
├── app.py                 # Flask web application
├── predictor.py          # Prediction service
├── model_trainer.py      # Model training script
├── templates/            # HTML templates
│   ├── index.html
│   └── predict.html
├── static/              # CSS and JS files
│   └── styles.css
└── requirements.txt     # Python dependencies
```

## Requirements

- Python 3.8+
- See `requirements.txt` for full dependencies
