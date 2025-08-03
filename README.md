# SMS Spam Detection System

A machine learning-powered web application that classifies SMS messages as spam or legitimate with 98.6% accuracy.

## Features

- **Real-time Classification**: Web interface for instant spam detection
- **Advanced ML Pipeline**: XGBoost classifier with engineered features
- **REST API**: Flask-based API for programmatic access
- **High Accuracy**: 98.6% accuracy with 0.97 F1-score
- **Feature Engineering**: 10+ features including URL/email detection, character ratios, and linguistic patterns

## Architecture

The system consists of modular components:
1. **Training Module** (`training.py`): Trains the XGBoost classifier using feature engineering pipeline
2. **Prediction Service** (`prediction.py`): Encapsulates model loading and inference logic
3. **Feature Engineering** (`feature_engineering.py`): Text preprocessing and feature extraction
4. **Web Application** (`app.py`): Flask web interface for real-time predictions
5. **CLI Interface** (`cli.py`): Command-line tool for interactive predictions

## Installation

```bash
# Clone the repository
git clone https://github.com/adityakamath255/sms-spam-identifier.git
cd sms-spam-identifier

# Install dependencies
pip install -r requirements.txt

# Download database and auxiliary data
python setup.py
```

## Usage

### Training a Model

```bash
python training.py
```

### Using the Command Line Interface

```bash
python cli.py
```

### Running the Web Application

```bash
python app.py
```

Navigate to `http://localhost:5000` to use the web interface.

## Dataset

The dataset used is [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)

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
├── app.py                    # Flask web application
├── cli.py                    # Command-line interface
├── prediction.py             # Prediction service and model loading
├── training.py               # Model training pipeline
├── feature_engineering.py    # Text preprocessing and feature extraction
├── setup.py                  # Setup script for external data
├── templates/                # HTML templates
│   ├── index.html
│   └── predict.html
├── static/                   # CSS and JS files
│   └── styles.css
└── requirements.txt          # Python dependencies
```

## Requirements

- Python 3.8+
- See `requirements.txt` for full dependencies
