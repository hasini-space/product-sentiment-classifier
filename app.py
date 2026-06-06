import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from src.text_cleaner import clean_text

# 1. Page Configuration
st.set_page_config(
    page_title="E-Commerce Sentiment Analyzer",
    page_icon="🛍️",
    layout="centered"
)

# 2. Title and Description
st.title("🛍️ E-Commerce Product Sentiment Classifier")
st.markdown("""
This interactive dashboard uses a Natural Language Processing (NLP) model to predict whether a customer review is **Positive** or **Negative**. 
""")

# 3. Cached Setup Function (So it only trains once when you launch the app)
@st.cache_resource
def load_trained_pipeline():
    # Load dataset
    df = pd.read_csv("https://github.com/Censius-AI/ECommerce-Women-Clothing-Reviews/raw/main/Womens%20Clothing%20E-Commerce%20Reviews.csv")
    
    # Store original data for visualization before filtering neutral reviews
    visual_df = df.copy()

    # Preprocess data for training
    df = df[['Review Text', 'Rating']].dropna(subset=['Review Text'])
    df = df[df['Rating'] != 3]  # Filter out neutral rating (3) for training
    df['Sentiment'] = df['Rating'].apply(lambda x: 1 if x > 3 else 0)
    
    # Process text
    df['Cleaned_Review'] = df['Review Text'].apply(clean_text)
    
    # Vectorize and Train
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['Cleaned_Review'])
    y = df['Sentiment']
    
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X, y)
    
    return vectorizer, model, visual_df

# Initialize the pipeline and get original data for visualization
with st.spinner("🧠 Loading NLP Model pipeline... Hang tight!"):
    vectorizer, model, visual_df = load_trained_pipeline()

st.success("✅ Model loaded and ready!")

st.divider()

# 4. Data Visualization Section
st.subheader("📊 Data Exploration: Star Rating Distribution")
st.markdown("""
Explore the distribution of star ratings in the original dataset. This helps understand the balance of positive, neutral, and negative reviews.
""")

# Calculate star counts and percentages
star_counts = visual_df['Rating'].value_counts().sort_index().reset_index()
star_counts.columns = ['Rating', 'Count']
total_reviews = star_counts['Count'].sum()
star_counts['Percentage'] = (star_counts['Count'] / total_reviews * 100).round(2)

# Create a bar chart using plotly.express
fig = px.bar(
    star_counts, 
    x='Rating', 
    y='Count', 
    color='Rating',
    text='Percentage',
    labels={'Rating': 'Star Rating', 'Count': 'Number of Reviews'},
    title='Count and Percentage of Reviews by Star Rating',
    color_discrete_sequence=px.colors.sequential.Tealgrn # Customize color scale
)
fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
fig.update_layout(showlegend=False)

st.plotly_chart(fig)

st.divider()

# 5. Interactive UI Elements
st.subheader("✍️ Test a Custom Review")
user_input = st.text_area(
    "Type a clothing or product review below:",
    placeholder="I bought this dress last week. The fabric feels cheap and the sizing is completely wrong..."
)

if st.button("Analyze Sentiment", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        # Preprocess user input
        cleaned_input = clean_text(user_input)
        
        # Transform using cached vectorizer
        vectorized_input = vectorizer.transform([cleaned_input])
        
        # Predict Class and Probability
        prediction = model.predict(vectorized_input)[0]
        probabilities = model.predict_proba(vectorized_input)[0]
        
        st.markdown("### 📊 Prediction Result:")
        
        if prediction == 1:
            st.balloons()
            st.success(f"**Positive Sentiment Detected!** (Confidence: {probabilities[1]:.2%})")
        else:
            st.error(f"**Negative Sentiment Detected!** (Confidence: {probabilities[0]:.2%})")
            
        # Display underlying metrics cleanly
        st.progress(float(probabilities[1]))
        st.caption("Sentiment slider scale: Left (0% Negative) to Right (100% Positive)")

# 6. Sidebar Analytics Dashboard
st.sidebar.header("📈 Dataset Quick Stats")
st.sidebar.markdown(f"""
* **Total Reviews Processed:** {len(visual_df)}
* **Algorithm:** Logistic Regression
* **Feature Extraction:** TF-IDF
* **Target Classes:** Binary (Positive / Negative)
* **Model Accuracy (from last run):** ~91%
""")
