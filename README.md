# E-Commerce Sentiment Analysis Dashboard

A professional-grade analytics dashboard that performs sentiment analysis on e-commerce reviews using state-of-the-art NLP models from Hugging Face.

## Features

### Products Section
- Interactive product catalog with search and filtering
- Category-based organization
- Key metrics: total products, categories, average ratings, stock status
- Sortable and filterable data table

### Testimonials Section
- Beautiful card-based testimonial display
- Customer information with locations and ratings
- Featured testimonial highlighting
- Date range filtering

### Reviews Section
- **Editable Date Range**: Interactive date pickers with preset options (All 2023, Q1-Q4)
- **Enhanced Month Selector**: Clear slider with month labels
- **Sentiment Analysis**: Powered by DistilBERT (distilbert-base-uncased-finetuned-sst-2-english)
- **Confidence Score Display**: Prominent display of average confidence scores for each sentiment
- **Real-time Metrics**:
  - Total reviews analyzed
  - Positive/Negative percentages with confidence scores
  - Average model certainty
- **Interactive Visualizations**:
  - Sentiment distribution bar chart with confidence labels
  - Confidence score histogram
  - Rating distribution
  - Sentiment vs. Star Rating correlation
  - Yearly trend analysis
  - **Word Cloud**: Visual representation of common words in positive/negative reviews
- **Smart Filtering**: Filter by sentiment, rating, and sort options
- **Export**: Download analyzed reviews as CSV or Excel

### AI Insights Section (NEW!)
- **Product Intelligence**:
  - AI-powered analysis of product reviews
  - Sentiment overview with confidence scores
  - Key topics discussed
  - What customers love (praise points)
  - Areas for improvement (complaint points)
  - AI-generated recommendations
  - Product performance comparison chart
- **Customer Insights Dashboard**:
  - Testimonial analysis with sentiment trends
  - Key themes identification
  - Satisfaction drivers
  - Topic distribution visualization
  - Improvement priorities
  - Common phrases display

## Tech Stack

- **Frontend**: Streamlit with custom CSS
- **NLP/ML**: Hugging Face Transformers (DistilBERT)
- **Visualization**: Plotly, WordCloud
- **Data Processing**: Pandas, NumPy
- **Export**: openpyxl (Excel), CSV
- **Backend**: Python 3.9+

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd data_mining
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Generate data** (if not already present):
   ```bash
   python scraper.py
   ```

5. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## Project Structure

```
data_mining/
├── app.py                  # Main Streamlit application
├── scraper.py              # Real web scraping + data generation
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── data/
│   ├── products.json       # Product catalog data
│   ├── reviews.json        # Customer reviews (2023)
│   └── testimonials.json   # Customer testimonials
└── utils/
    ├── __init__.py         # Package initialization
    ├── data_loader.py      # Data loading utilities
    ├── sentiment.py        # Sentiment analysis module
    ├── visualizations.py   # Plotly chart functions + word cloud
    └── ai_insights.py      # AI-powered insights generation
```

## Usage Guide

### Navigating the Dashboard

1. **Sidebar Navigation**: Use the sidebar to switch between:
   - Products
   - Testimonials
   - Reviews
   - AI Insights (NEW!)

2. **Date Range Selection**:
   - Choose from preset ranges (All 2023, Q1-Q4)
   - Or select custom start/end dates
   - All data is filtered based on selected range

3. **Products**: Search for products or filter by category

4. **Testimonials**: Scroll through customer testimonials (filtered by date)

5. **Reviews**:
   - Use the enhanced month slider to select a time period
   - View sentiment analysis metrics with confidence scores
   - Explore visualizations in different tabs (including word cloud!)
   - Filter and sort reviews
   - Export data to CSV or Excel

6. **AI Insights** (NEW!):
   - Select a product to get AI-powered analysis
   - View praise points and improvement areas
   - Get actionable AI recommendations
   - Explore customer insight dashboard

### Sentiment Analysis

The application uses DistilBERT fine-tuned on SST-2 for binary sentiment classification:
- **POSITIVE**: Reviews expressing satisfaction, praise, or positive experiences
- **NEGATIVE**: Reviews expressing dissatisfaction, complaints, or negative experiences

Each prediction includes a confidence score (0.5 - 1.0) indicating model certainty, displayed prominently in:
- Sentiment overview cards
- Bar chart labels
- Individual review cards

### Web Scraping

The scraper attempts to collect real data from https://web-scraping.dev/:
- **Products**: Scrapes all pages with pagination handling
- **Reviews**: Uses GraphQL API with cursor-based pagination
- **Testimonials**: Handles infinite scroll/HTMX pagination

If scraping fails (e.g., network restrictions), it generates realistic synthetic data.

## Performance Optimizations

- **Caching**: Streamlit's `@st.cache_data` and `@st.cache_resource` decorators
- **Batch Processing**: Reviews are processed in batches for efficiency
- **Pre-loaded Data**: JSON data files for instant loading
- **Lazy Model Loading**: Transformer model loaded only when needed
- **Fallback Analyzer**: Rule-based sentiment analysis when transformer unavailable

## Data Description

### Products
- Multiple product categories
- Fields: id, name, category, price, rating, review_count, in_stock, date_added, description

### Reviews
- Distributed across all 12 months of 2023
- Fields: id, product_id, product_name, category, rating (1-5), text, date, verified_purchase, helpful_votes

### Testimonials
- Customer success stories and feedback
- Fields: id, text, author, date, rating, location, featured

## Customization

### Styling
Modify the `load_custom_css()` function in `app.py` to change:
- Color scheme
- Card styles
- Typography
- Layout

### Colors
Edit the `COLORS` dictionary in `utils/visualizations.py`:
```python
COLORS = {
    'positive': '#10B981',  # Green
    'negative': '#EF4444',  # Red
    'primary': '#3B82F6',   # Blue
    ...
}
```

### AI Insights
Customize the `AIInsightsAnalyzer` class in `utils/ai_insights.py`:
- Add/modify topic keywords
- Adjust sentiment keyword weights
- Customize recommendation templates

## Deployment

### Streamlit Cloud
1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Deploy with one click

### Render
1. Create a new Web Service
2. Connect GitHub repository
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `streamlit run app.py --server.port $PORT`

## New in This Version

- Editable date range filter with presets
- Enhanced month slider with clear labels
- Confidence scores displayed on all visualizations
- Word cloud visualization for positive/negative reviews
- AI-powered product intelligence with recommendations
- AI-powered testimonial/customer insights
- Excel export functionality
- Real web scraping with fallback to synthetic data

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for the transformers library
- [Streamlit](https://streamlit.io/) for the amazing framework
- [Plotly](https://plotly.com/) for interactive visualizations
- [WordCloud](https://amueller.github.io/word_cloud/) for word cloud generation
