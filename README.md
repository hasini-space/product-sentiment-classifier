# 🛍️ E-Commerce Product Sentiment Classifier

An interactive Streamlit dashboard that uses Natural Language Processing (NLP) and machine learning to predict whether customer reviews are **Positive** or **Negative**.

## Features

- 📊 Real-time sentiment analysis using Logistic Regression
- 🎨 Interactive visualizations with Plotly
- ✍️ Test custom reviews and get confidence scores
- 📈 Dataset statistics and insights
- 🚀 Fast inference with TF-IDF vectorization

## Setup Instructions

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/hasini-space/product-sentiment-classifier.git
cd product-sentiment-classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Dataset

The application uses the **Women's Clothing E-Commerce Reviews dataset**, which contains customer reviews and ratings from an e-commerce platform.

### Dataset Source

The dataset is automatically downloaded from:
```
https://raw.githubusercontent.com/Censius-AI/ECommerce-Women-Clothing-Reviews/main/Womens%20Clothing%20E-Commerce%20Reviews.csv
```

**Dataset Size:** ~8.5 MB CSV file (not included in this repository due to size constraints)

## Troubleshooting

### HTTP Error When Running the App

If you encounter an `urllib.error.HTTPError` when the app tries to load the dataset, this is likely due to:

1. **Network/Firewall restrictions** - The Streamlit Cloud environment or your network may block access to raw GitHub URLs
2. **Rate limiting** - GitHub may rate-limit requests from certain IP addresses
3. **Repository unavailability** - The source repository may be temporarily unavailable

#### Solutions

**Option 1: Download and Add Locally (Recommended for Development)**
1. Download the dataset from [Censius-AI/ECommerce-Women-Clothing-Reviews](https://github.com/Censius-AI/ECommerce-Women-Clothing-Reviews)
2. Create a `data/` folder in your project root
3. Place the CSV file: `data/Womens_Clothing_E-Commerce_Reviews.csv`
4. Modify line 25 in `app.py`:
```python
df = pd.read_csv("data/Womens_Clothing_E-Commerce_Reviews.csv")
```

**Option 2: Use Alternative Dataset Source**
If the primary source fails, you can try:
- [abdelrahmansamir1/Women-s-E-Commerce-Clothing-Reviews](https://github.com/abdelrahmansamir1/Women-s-E-Commerce-Clothing-Reviews)
- [Kaggle: Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets)

**Option 3: Deploy with Secrets**
On Streamlit Cloud:
1. Upload the CSV file to GitHub (in a private branch or use Git LFS)
2. Use GitHub API with authentication to bypass rate limits
3. Set `GITHUB_TOKEN` as a secret in Streamlit Cloud settings

## Project Structure

```
product-sentiment-classifier/
├── app.py                      # Main Streamlit application
├── src/
│   └── text_cleaner.py        # Text preprocessing utilities
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── data/                       # (Optional) Directory for local CSV file
    └── Womens_Clothing_E-Commerce_Reviews.csv
```

## Model Details

- **Algorithm:** Logistic Regression
- **Feature Extraction:** TF-IDF Vectorizer (5000 features)
- **Training Data:** Binary classification (Positive/Negative)
- **Accuracy:** ~91%
- **Preprocessing:** Custom text cleaning (lowercasing, tokenization, stopword removal)

## Usage

1. Open the app in your browser
2. Explore the "📊 Data Exploration" section to see review distributions
3. Enter a product review in the "✍️ Test a Custom Review" section
4. Click "Analyze Sentiment" to get a prediction
5. View the confidence score and sentiment slider

## Dependencies

- streamlit
- pandas
- plotly
- scikit-learn
- (see requirements.txt for full list)

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Feel free to:
- Report issues
- Suggest improvements
- Submit pull requests

## Support

For questions or issues, please open a GitHub issue in this repository.

---

**Note:** This application requires internet connectivity to download the dataset on first run. If you experience persistent HTTP errors, use Option 1 above (local dataset) for the most reliable setup.
