"""
E-Commerce Sentiment Analysis Dashboard

A professional-grade analytics dashboard that provides sentiment analysis
for e-commerce reviews using Hugging Face transformers.

Features:
- Real-time sentiment analysis
- AI-powered product insights
- Interactive date filtering
- Word cloud visualizations
- Export functionality

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
from datetime import datetime, date
from io import BytesIO

from utils.data_loader import DataLoader, validate_data_files
from utils.sentiment import SentimentAnalyzer, get_sentiment_summary, format_sentiment_label
from utils.visualizations import (
    create_sentiment_chart,
    create_confidence_histogram,
    create_trend_chart,
    create_rating_distribution_chart,
    create_category_chart,
    create_sentiment_by_rating_chart,
    create_word_cloud,
    create_comparison_chart,
    create_confidence_trend_chart,
    create_product_performance_chart,
    extract_common_phrases,
    COLORS,
    WORDCLOUD_AVAILABLE,
)
from utils.ai_insights import AIInsightsAnalyzer, get_product_insights, get_testimonial_insights


# Page configuration
st.set_page_config(
    page_title="E-Commerce Sentiment Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_custom_css():
    """Load custom CSS styling for the application."""
    st.markdown("""
    <style>
        /* Main app styling */
        .main {
            background-color: #0F172A;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #1E293B;
            border-right: 1px solid #334155;
        }

        [data-testid="stSidebar"] .block-container {
            padding-top: 2rem;
        }

        /* Headers */
        h1, h2, h3 {
            color: #F1F5F9 !important;
        }

        /* Cards */
        .metric-card {
            background: linear-gradient(135deg, #1E293B 0%, #334155 100%);
            border: 1px solid #475569;
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        }

        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            color: #F1F5F9;
            margin: 0;
        }

        .metric-label {
            font-size: 0.875rem;
            color: #94A3B8;
            margin-top: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .metric-delta-positive {
            color: #10B981;
            font-size: 0.875rem;
        }

        .metric-delta-negative {
            color: #EF4444;
            font-size: 0.875rem;
        }

        /* Review cards */
        .review-card {
            background-color: #1E293B;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1rem;
            transition: transform 0.2s, box-shadow 0.2s;
        }

        .review-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px -5px rgba(0, 0, 0, 0.4);
        }

        .review-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.75rem;
        }

        .review-product {
            font-weight: 600;
            color: #E2E8F0;
            font-size: 1rem;
        }

        .review-date {
            color: #64748B;
            font-size: 0.875rem;
        }

        .review-text {
            color: #CBD5E1;
            line-height: 1.6;
            margin: 0.75rem 0;
        }

        .review-footer {
            display: flex;
            gap: 1rem;
            align-items: center;
            margin-top: 0.75rem;
        }

        .sentiment-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }

        .sentiment-positive {
            background-color: rgba(16, 185, 129, 0.2);
            color: #10B981;
            border: 1px solid #10B981;
        }

        .sentiment-negative {
            background-color: rgba(239, 68, 68, 0.2);
            color: #EF4444;
            border: 1px solid #EF4444;
        }

        .confidence-score {
            color: #94A3B8;
            font-size: 0.875rem;
        }

        .rating-stars {
            color: #FBBF24;
        }

        /* Testimonial cards */
        .testimonial-card {
            background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
            border: 1px solid #334155;
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            position: relative;
        }

        .testimonial-quote {
            font-size: 3rem;
            color: #3B82F6;
            position: absolute;
            top: 0.5rem;
            left: 1rem;
            opacity: 0.3;
        }

        .testimonial-text {
            color: #E2E8F0;
            font-style: italic;
            line-height: 1.8;
            margin-left: 1.5rem;
        }

        .testimonial-author {
            color: #94A3B8;
            margin-top: 1rem;
            text-align: right;
        }

        /* Product table styling */
        .dataframe {
            background-color: #1E293B !important;
        }

        /* Section headers */
        .section-header {
            background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 1.75rem;
            font-weight: 700;
            margin-bottom: 1.5rem;
        }

        /* Month selector */
        .month-display {
            background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%);
            border-radius: 12px;
            padding: 1rem 2rem;
            text-align: center;
            margin-bottom: 1.5rem;
        }

        .month-display h2 {
            margin: 0;
            font-size: 1.5rem;
        }

        /* Date range display */
        .date-range-display {
            background: linear-gradient(135deg, #10B981 0%, #3B82F6 100%);
            border-radius: 12px;
            padding: 0.75rem 1.5rem;
            text-align: center;
            margin-bottom: 1rem;
        }

        /* Info boxes */
        .info-box {
            background-color: rgba(59, 130, 246, 0.1);
            border: 1px solid #3B82F6;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }

        /* AI Insight cards */
        .insight-card {
            background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
            border: 1px solid #475569;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
        }

        .insight-title {
            color: #3B82F6;
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 0.75rem;
        }

        .insight-list {
            color: #CBD5E1;
            padding-left: 1.5rem;
        }

        .insight-list li {
            margin-bottom: 0.5rem;
        }

        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        ::-webkit-scrollbar-track {
            background: #1E293B;
        }

        ::-webkit-scrollbar-thumb {
            background: #475569;
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: #64748B;
        }

        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: #1E293B;
            border-radius: 8px;
        }

        /* Select slider */
        .stSlider > div > div > div {
            background-color: #3B82F6;
        }
    </style>
    """, unsafe_allow_html=True)


def render_sidebar(data_loader: DataLoader) -> tuple[str, date, date]:
    """
    Render the sidebar navigation and return selected section and date range.

    Args:
        data_loader: DataLoader instance for statistics

    Returns:
        Tuple of (selected section, start date, end date)
    """
    with st.sidebar:
        # Logo and title
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="font-size: 1.5rem; margin-bottom: 0.25rem;">
                📊 Sentiment Analytics
            </h1>
            <p style="color: #64748B; font-size: 0.875rem;">
                E-Commerce Intelligence Dashboard
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Section selector
        st.markdown("### Navigation")

        section = st.radio(
            "Select Section",
            options=["Products", "Testimonials", "Reviews", "AI Insights"],
            label_visibility="collapsed",
            key="section_selector"
        )

        st.markdown("---")

        # Year Filter
        st.markdown("### Year Selection")

        # Get available years from data
        available_years = data_loader.get_available_years()
        default_index = len(available_years) - 1 if available_years else 0

        selected_year = st.selectbox(
            "Select Year",
            options=available_years,
            index=default_index,
            key="year_selector"
        )

        # Set date range for the entire year
        start_date = date(selected_year, 1, 1)
        end_date = date(selected_year, 12, 31)

        st.markdown("---")

        # Dataset statistics
        st.markdown("### Dataset Overview")

        try:
            stats = data_loader.get_data_statistics()

            st.markdown(f"""
            <div style="background-color: #0F172A; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                <p style="color: #94A3B8; margin: 0; font-size: 0.75rem;">PRODUCTS</p>
                <p style="color: #F1F5F9; margin: 0; font-size: 1.25rem; font-weight: 600;">{stats['total_products']}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background-color: #0F172A; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                <p style="color: #94A3B8; margin: 0; font-size: 0.75rem;">REVIEWS</p>
                <p style="color: #F1F5F9; margin: 0; font-size: 1.25rem; font-weight: 600;">{stats['total_reviews']}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background-color: #0F172A; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                <p style="color: #94A3B8; margin: 0; font-size: 0.75rem;">TESTIMONIALS</p>
                <p style="color: #F1F5F9; margin: 0; font-size: 1.25rem; font-weight: 600;">{stats['total_testimonials']}</p>
            </div>
            """, unsafe_allow_html=True)

            # Show selected year
            st.markdown(f"""
            <div style="background-color: #0F172A; padding: 1rem; border-radius: 8px;">
                <p style="color: #94A3B8; margin: 0; font-size: 0.75rem;">SELECTED YEAR</p>
                <p style="color: #F1F5F9; margin: 0; font-size: 1.5rem; font-weight: 600;">
                    {selected_year}
                </p>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error loading stats: {str(e)}")

        st.markdown("---")

        # Info section
        st.markdown("""
        <div style="color: #64748B; font-size: 0.75rem; text-align: center;">
            <p>Powered by Hugging Face Transformers</p>
            <p>Model: DistilBERT SST-2</p>
        </div>
        """, unsafe_allow_html=True)

    return section, start_date, end_date


def render_products_section(data_loader: DataLoader):
    """Render the Products section."""
    st.markdown('<p class="section-header">Product Catalog</p>', unsafe_allow_html=True)

    try:
        products = data_loader.load_products()

        # Metrics row
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Products", len(products))

        with col2:
            categories = products['category'].nunique()
            st.metric("Categories", categories)

        with col3:
            avg_rating = products['rating'].mean()
            st.metric("Avg Rating", f"{avg_rating:.1f} ⭐")

        st.markdown("---")

        # Filters
        col1, col2 = st.columns([2, 2])

        with col1:
            search = st.text_input("🔍 Search products", placeholder="Enter product name...")

        with col2:
            categories_list = ["All"] + sorted(products['category'].unique().tolist())
            selected_category = st.selectbox("Filter by Category", categories_list)

        # Apply filters
        filtered = products.copy()

        if search:
            filtered = filtered[filtered['name'].str.contains(search, case=False, na=False)]

        if selected_category != "All":
            filtered = filtered[filtered['category'] == selected_category]

        st.markdown(f"**Showing {len(filtered)} products**")

        # Display dataframe
        display_cols = ['name', 'category', 'price', 'rating', 'review_count']
        st.dataframe(
            filtered[display_cols].rename(columns={
                'name': 'Product Name',
                'category': 'Category',
                'price': 'Price ($)',
                'rating': 'Rating',
                'review_count': 'Reviews'
            }),
            use_container_width=True,
            height=500
        )

    except Exception as e:
        st.error(f"Error loading products: {str(e)}")


def render_testimonials_section(data_loader: DataLoader, start_date: date, end_date: date):
    """Render the Testimonials section."""
    st.markdown('<p class="section-header">Customer Testimonials</p>', unsafe_allow_html=True)

    try:
        testimonials = data_loader.load_testimonials()

        # Filter by date range if date column exists
        if 'date' in testimonials.columns:
            testimonials['date'] = pd.to_datetime(testimonials['date'])
            filtered_testimonials = testimonials[
                (testimonials['date'].dt.date >= start_date) &
                (testimonials['date'].dt.date <= end_date)
            ]
        else:
            # No date filtering if dates don't exist
            filtered_testimonials = testimonials

        # Metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Testimonials", len(filtered_testimonials))

        with col2:
            avg_rating = filtered_testimonials['rating'].mean() if not filtered_testimonials.empty else 0
            st.metric("Avg Rating", f"{avg_rating:.1f} ⭐")

        with col3:
            avg_length = filtered_testimonials['text'].str.len().mean() if not filtered_testimonials.empty else 0
            st.metric("Avg Length", f"{avg_length:.0f} chars")

        with col4:
            featured = filtered_testimonials['featured'].sum() if 'featured' in filtered_testimonials.columns else 0
            st.metric("Featured", featured)

        st.markdown("---")

        # Display testimonials as cards
        if filtered_testimonials.empty:
            st.info("No testimonials found for the selected date range")
        else:
            for _, row in filtered_testimonials.iterrows():
                # Create star rating
                stars = "⭐" * int(row['rating'])

                st.markdown(f"""
                <div class="testimonial-card">
                    <span class="testimonial-quote">"</span>
                    <p class="testimonial-text">{row['text']}</p>
                    <p class="testimonial-author">
                        <span class="rating-stars">{stars}</span>
                    </p>
                </div>
                """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error loading testimonials: {str(e)}")


def render_reviews_section(
    data_loader: DataLoader,
    analyzer: SentimentAnalyzer,
    start_date: date,
    end_date: date
):
    """Render the Reviews section with sentiment analysis."""
    st.markdown('<p class="section-header">Review Sentiment Analysis</p>', unsafe_allow_html=True)

    try:
        # Display selected year
        st.markdown(f"""
        <div class="date-range-display">
            <span style="color: white; font-weight: 600;">
                📅 Year: {start_date.year}
            </span>
        </div>
        """, unsafe_allow_html=True)

        # Month range selector with slider
        months = [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]

        st.markdown("### Select Month Range")

        # Range slider for month selection
        month_range = st.slider(
            "Month Range",
            min_value=1,
            max_value=12,
            value=(1, 12),
            format="",
            key="reviews_month_range",
            help="Drag to select the month range for analysis"
        )

        start_month_num = month_range[0]
        end_month_num = month_range[1]
        start_month_name = months[start_month_num - 1]
        end_month_name = months[end_month_num - 1]

        # Display selected range
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 100%); padding: 0.75rem 1.5rem; border-radius: 8px; text-align: center; margin-bottom: 1rem;">
            <span style="color: white; font-weight: 600;">📅 {start_month_name} - {end_month_name} {start_date.year}</span>
        </div>
        """, unsafe_allow_html=True)

        # Get reviews for selected month range
        all_reviews_data = data_loader.load_reviews()

        # Filter by month range (handle wrap-around if end < start)
        if start_month_num <= end_month_num:
            reviews = all_reviews_data[
                (all_reviews_data['month'] >= start_month_num) &
                (all_reviews_data['month'] <= end_month_num)
            ].copy()
        else:
            # Handle year wrap-around (e.g., November to February)
            reviews = all_reviews_data[
                (all_reviews_data['month'] >= start_month_num) |
                (all_reviews_data['month'] <= end_month_num)
            ].copy()

        # Filter by date range
        reviews['date'] = pd.to_datetime(reviews['date'])
        reviews = reviews[
            (reviews['date'].dt.date >= start_date) &
            (reviews['date'].dt.date <= end_date)
        ]

        if reviews.empty:
            st.warning(f"No reviews found for {start_month_name} - {end_month_name} in the selected date range")
            return

        # Perform sentiment analysis
        with st.spinner("Analyzing sentiments..."):
            analyzed_reviews = analyzer.analyze_dataframe(reviews, text_column='text')

        # Get summary statistics
        summary = get_sentiment_summary(analyzed_reviews)

        # Metrics row with confidence scores prominently displayed
        st.markdown("### Sentiment Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value">{summary['total']}</p>
                <p class="metric-label">Total Reviews</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value" style="color: #10B981;">{summary['positive_pct']:.1f}%</p>
                <p class="metric-label">Positive</p>
                <p class="metric-delta-positive">({summary['positive_count']} reviews)</p>
                <p style="color: #94A3B8; font-size: 0.75rem;">Avg Conf: {summary.get('positive_avg_conf', summary['avg_confidence']):.0%}</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value" style="color: #EF4444;">{summary['negative_pct']:.1f}%</p>
                <p class="metric-label">Negative</p>
                <p class="metric-delta-negative">({summary['negative_count']} reviews)</p>
                <p style="color: #94A3B8; font-size: 0.75rem;">Avg Conf: {summary.get('negative_avg_conf', summary['avg_confidence']):.0%}</p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value">{summary['avg_confidence']:.0%}</p>
                <p class="metric-label">Avg Confidence</p>
                <p style="color: #94A3B8; font-size: 0.75rem;">Model certainty</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Visualizations
        st.markdown("### Visualizations")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Sentiment Distribution",
            "📈 Confidence Scores",
            "⭐ By Rating",
            "📅 Yearly Trend",
            "☁️ Word Cloud"
        ])

        with tab1:
            fig = create_sentiment_chart(analyzed_reviews, f"Sentiment Distribution - {start_month_name} to {end_month_name} {start_date.year}")
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            fig = create_confidence_histogram(analyzed_reviews)
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                fig = create_rating_distribution_chart(analyzed_reviews)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = create_sentiment_by_rating_chart(analyzed_reviews)
                st.plotly_chart(fig, use_container_width=True)

        with tab4:
            # Calculate yearly trend data
            all_reviews = data_loader.load_reviews()
            all_reviews['date'] = pd.to_datetime(all_reviews['date'])
            all_reviews = all_reviews[
                (all_reviews['date'].dt.date >= start_date) &
                (all_reviews['date'].dt.date <= end_date)
            ]

            with st.spinner("Calculating yearly trends..."):
                all_analyzed = analyzer.analyze_dataframe(all_reviews, text_column='text')

            monthly_data = {}
            for month in months:
                month_num = months.index(month) + 1
                month_reviews = all_analyzed[all_analyzed['month'] == month_num]
                if not month_reviews.empty:
                    month_summary = get_sentiment_summary(month_reviews)
                    monthly_data[month] = month_summary

            fig = create_trend_chart(monthly_data)
            st.plotly_chart(fig, use_container_width=True)

        with tab5:
            # Word Cloud visualization
            if WORDCLOUD_AVAILABLE:
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### Positive Reviews")
                    positive_texts = analyzed_reviews[
                        analyzed_reviews['sentiment_label'] == 'POSITIVE'
                    ]['text'].tolist()

                    if positive_texts:
                        wc_positive = create_word_cloud(positive_texts, 'positive')
                        if wc_positive:
                            st.image(wc_positive, use_container_width=True)
                        else:
                            st.info("Could not generate word cloud")
                    else:
                        st.info("No positive reviews to display")

                with col2:
                    st.markdown("#### Negative Reviews")
                    negative_texts = analyzed_reviews[
                        analyzed_reviews['sentiment_label'] == 'NEGATIVE'
                    ]['text'].tolist()

                    if negative_texts:
                        wc_negative = create_word_cloud(negative_texts, 'negative')
                        if wc_negative:
                            st.image(wc_negative, use_container_width=True)
                        else:
                            st.info("Could not generate word cloud")
                    else:
                        st.info("No negative reviews to display")
            else:
                st.info("Word cloud requires the 'wordcloud' package. Install with: pip install wordcloud")

                # Fallback: show common phrases
                st.markdown("#### Common Phrases")
                all_texts = analyzed_reviews['text'].tolist()
                phrases = extract_common_phrases(all_texts, top_n=15)

                if phrases:
                    phrase_df = pd.DataFrame(phrases, columns=['Word', 'Count'])
                    st.dataframe(phrase_df, use_container_width=True)

        st.markdown("---")

        # Reviews list
        st.markdown("### Reviews")

        # Filter options
        col1, col2, col3 = st.columns([2, 2, 2])

        with col1:
            sentiment_filter = st.selectbox(
                "Filter by Sentiment",
                ["All", "Positive", "Negative"]
            )

        with col2:
            rating_filter = st.selectbox(
                "Filter by Rating",
                ["All", "5 Stars", "4 Stars", "3 Stars", "2 Stars", "1 Star"]
            )

        with col3:
            sort_by = st.selectbox(
                "Sort by",
                ["Date (Newest)", "Date (Oldest)", "Confidence (High)", "Confidence (Low)"]
            )

        # Apply filters
        filtered_reviews = analyzed_reviews.copy()

        if sentiment_filter != "All":
            filtered_reviews = filtered_reviews[
                filtered_reviews['sentiment_label'] == sentiment_filter.upper()
            ]

        if rating_filter != "All":
            rating = int(rating_filter.split()[0])
            filtered_reviews = filtered_reviews[filtered_reviews['rating'] == rating]

        # Apply sorting
        if sort_by == "Date (Newest)":
            filtered_reviews = filtered_reviews.sort_values('date', ascending=False)
        elif sort_by == "Date (Oldest)":
            filtered_reviews = filtered_reviews.sort_values('date', ascending=True)
        elif sort_by == "Confidence (High)":
            filtered_reviews = filtered_reviews.sort_values('sentiment_score', ascending=False)
        else:
            filtered_reviews = filtered_reviews.sort_values('sentiment_score', ascending=True)

        st.markdown(f"**Showing {len(filtered_reviews)} reviews**")

        # Display reviews (limit to first 20 for performance)
        for _, row in filtered_reviews.head(20).iterrows():
            sentiment_class = "sentiment-positive" if row['sentiment_label'] == 'POSITIVE' else "sentiment-negative"

            st.markdown(f"""
            <div class="review-card">
                <div class="review-header">
                    <span class="review-product">{row['product_name']}</span>
                    <span class="review-date">{row['date'].strftime('%B %d, %Y')}</span>
                </div>
                <div class="rating-stars">{"⭐" * int(row['rating'])}</div>
                <p class="review-text">{row['text']}</p>
                <div class="review-footer">
                    <span class="sentiment-badge {sentiment_class}">{format_sentiment_label(row['sentiment_label'])}</span>
                    <span class="confidence-score">Confidence: {row['sentiment_score']:.0%}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if len(filtered_reviews) > 20:
            st.info(f"Showing 20 of {len(filtered_reviews)} reviews. Export to see all.")

        # Export options
        st.markdown("---")
        st.markdown("### Export Data")

        col1, col2 = st.columns(2)

        with col1:
            csv = filtered_reviews.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"reviews_{start_month_name.lower()}_to_{end_month_name.lower()}_{start_date.year}_analyzed.csv",
                mime="text/csv"
            )

        with col2:
            # Excel export
            try:
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    filtered_reviews.to_excel(writer, index=False, sheet_name='Reviews')
                buffer.seek(0)

                st.download_button(
                    label="📊 Download Excel",
                    data=buffer,
                    file_name=f"reviews_{start_month_name.lower()}_to_{end_month_name.lower()}_{start_date.year}_analyzed.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception:
                st.info("Install openpyxl for Excel export: pip install openpyxl")

    except Exception as e:
        st.error(f"Error in reviews section: {str(e)}")
        st.exception(e)


def render_ai_insights_section(
    data_loader: DataLoader,
    analyzer: SentimentAnalyzer,
    start_date: date,
    end_date: date
):
    """Render the AI Insights section."""
    # Title and comparison year selector on the same line
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<p class="section-header">AI-Powered Insights</p>', unsafe_allow_html=True)
    with col2:
        # Get available years from data (excluding current year)
        available_years = data_loader.get_available_years()
        comparison_years = [y for y in available_years if y != start_date.year]
        if not comparison_years:
            comparison_years = available_years
        default_comparison_index = 0 if comparison_years else 0

        comparison_year = st.selectbox(
            "Compare to Year",
            options=comparison_years,
            index=default_comparison_index,
            key="comparison_year",
            help="Select a year to compare performance metrics against"
        )

    # Display analyzing year
    st.markdown(f"""
    <div class="date-range-display">
        <span style="color: white; font-weight: 600;">
            🤖 Analyzing: {start_date.year}
        </span>
    </div>
    """, unsafe_allow_html=True)

    try:
        # Load and analyze reviews
        all_reviews = data_loader.load_reviews()
        all_reviews['date'] = pd.to_datetime(all_reviews['date'])
        filtered_reviews = all_reviews[
            (all_reviews['date'].dt.date >= start_date) &
            (all_reviews['date'].dt.date <= end_date)
        ]

        with st.spinner("Performing sentiment analysis..."):
            analyzed_reviews = analyzer.analyze_dataframe(filtered_reviews, text_column='text')

        # Create tabs for different insights
        tab1, tab2 = st.tabs(["📦 Product Intelligence", "🎯 Customer Insights"])

        with tab1:
            st.markdown("### Product Performance Analysis")
            st.markdown("Select a product to see performance metrics and actionable suggestions.")

            # Product selector
            products = data_loader.load_products()
            product_names = ["All Products"] + sorted(products['name'].unique().tolist())

            selected_product = st.selectbox(
                "Select Product",
                options=product_names,
                key="product_selector"
            )

            if selected_product == "All Products":
                # Show overall product performance
                st.markdown("#### Overall Product Performance")

                fig = create_product_performance_chart(analyzed_reviews)
                st.plotly_chart(fig, use_container_width=True)

                # Overall summary
                summary = get_sentiment_summary(analyzed_reviews)
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Total Reviews", summary['total'])
                with col2:
                    st.metric("Positive Rate", f"{summary['positive_pct']:.1f}%")
                with col3:
                    st.metric("Avg Confidence", f"{summary['avg_confidence']:.0%}")

            else:
                # Get AI insights for selected product
                with st.spinner(f"Analyzing {selected_product}..."):
                    insights = get_product_insights(selected_product, analyzed_reviews)

                # Load comparison year data
                comparison_start = date(comparison_year, 1, 1)
                comparison_end = date(comparison_year, 12, 31)
                comparison_reviews = all_reviews[
                    (all_reviews['date'].dt.date >= comparison_start) &
                    (all_reviews['date'].dt.date <= comparison_end)
                ]

                # Get comparison insights
                comparison_insights = None
                if not comparison_reviews.empty:
                    with st.spinner(f"Analyzing comparison data for {comparison_year}..."):
                        comparison_analyzed = analyzer.analyze_dataframe(comparison_reviews, text_column='text')
                        comparison_insights = get_product_insights(selected_product, comparison_analyzed)

                # Performance Metrics
                st.markdown(f"#### 📊 Performance Metrics (Compared to {comparison_year})")
                col1, col2, col3 = st.columns(3)

                with col1:
                    if comparison_insights:
                        rating_delta = insights['avg_rating'] - comparison_insights['avg_rating']
                        st.metric("Average Rating", f"{insights['avg_rating']:.2f} ⭐",
                                 delta=f"{rating_delta:+.2f}")
                    else:
                        st.metric("Average Rating", f"{insights['avg_rating']:.2f} ⭐")

                with col2:
                    if comparison_insights:
                        positive_delta = insights['positive_count'] - comparison_insights['positive_count']
                        st.metric("Positive Reviews", insights['positive_count'],
                                 delta=f"{positive_delta:+d} vs {comparison_year}")
                    else:
                        st.metric("Positive Reviews", insights['positive_count'])

                with col3:
                    if comparison_insights:
                        negative_delta = insights['negative_count'] - comparison_insights['negative_count']
                        st.metric("Negative Reviews", insights['negative_count'],
                                 delta=f"{negative_delta:+d} vs {comparison_year}",
                                 delta_color="inverse")
                    else:
                        st.metric("Negative Reviews", insights['negative_count'])

                st.markdown("---")

                # Action Suggestions
                st.markdown("#### 💡 Action Suggestions to Improve Performance")
                for i, suggestion in enumerate(insights['action_suggestions'], 1):
                    st.markdown(f"""
                    <div style="background: #1E293B; padding: 1rem; border-radius: 8px; margin-bottom: 0.75rem; border-left: 4px solid #3B82F6;">
                        <span style="color: #E2E8F0; font-size: 0.95rem;">{suggestion}</span>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")

                # Important Phrases
                st.markdown("#### 💬 Most Important Customer Feedback")
                if insights['important_phrases']:
                    for phrase in insights['important_phrases']:
                        st.markdown(f"""
                        <div style="background: #0F172A; padding: 0.75rem 1rem; border-radius: 8px; margin-bottom: 0.5rem; border-left: 3px solid #10B981;">
                            <span style="color: #CBD5E1; font-style: italic;">"{phrase}"</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No significant phrases identified")

        with tab2:
            st.markdown("### Testimonials Performance Analysis")
            st.markdown("Overview of customer testimonials with actionable suggestions.")

            # Load testimonials
            testimonials = data_loader.load_testimonials()

            # Filter by date range if date column exists
            if 'date' in testimonials.columns:
                testimonials['date'] = pd.to_datetime(testimonials['date'])
                filtered_testimonials = testimonials[
                    (testimonials['date'].dt.date >= start_date) &
                    (testimonials['date'].dt.date <= end_date)
                ]
            else:
                # No date filtering if dates don't exist
                filtered_testimonials = testimonials

            with st.spinner("Analyzing testimonials..."):
                testimonial_insights = get_testimonial_insights(filtered_testimonials)

            # Performance Metrics (no year comparison)
            st.markdown("#### 📊 Performance Metrics")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Average Rating", f"{testimonial_insights['avg_rating']:.2f} ⭐")

            with col2:
                st.metric("Positive Testimonials", testimonial_insights['positive_count'])

            with col3:
                st.metric("Negative Testimonials", testimonial_insights['negative_count'],
                         delta_color="inverse")

            st.markdown("---")

            # Action Suggestions
            st.markdown("#### 💡 Action Suggestions to Improve Performance")
            for i, suggestion in enumerate(testimonial_insights['action_suggestions'], 1):
                st.markdown(f"""
                <div style="background: #1E293B; padding: 1rem; border-radius: 8px; margin-bottom: 0.75rem; border-left: 4px solid #8B5CF6;">
                    <span style="color: #E2E8F0; font-size: 0.95rem;">{suggestion}</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # Important Phrases
            st.markdown("#### 💬 Most Important Customer Testimonials")
            if testimonial_insights['important_phrases']:
                for phrase in testimonial_insights['important_phrases']:
                    st.markdown(f"""
                    <div style="background: #0F172A; padding: 0.75rem 1rem; border-radius: 8px; margin-bottom: 0.5rem; border-left: 3px solid #8B5CF6;">
                        <span style="color: #CBD5E1; font-style: italic;">"{phrase}"</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No significant phrases identified")

    except Exception as e:
        st.error(f"Error in AI insights section: {str(e)}")
        st.exception(e)


def main():
    """Main application entry point."""
    # Load custom CSS
    load_custom_css()

    # Validate data files
    validation = validate_data_files()

    if not all(validation.values()):
        st.error("Missing data files. Please run `python scraper.py` first.")
        missing = [f for f, exists in validation.items() if not exists]
        st.write(f"Missing files: {', '.join(missing)}")
        return

    # Initialize components
    data_loader = DataLoader()
    analyzer = SentimentAnalyzer()

    # Render sidebar and get selected section + date range
    section, start_date, end_date = render_sidebar(data_loader)

    # Render selected section
    if section == "Products":
        render_products_section(data_loader)
    elif section == "Testimonials":
        render_testimonials_section(data_loader, start_date, end_date)
    elif section == "Reviews":
        render_reviews_section(data_loader, analyzer, start_date, end_date)
    elif section == "AI Insights":
        render_ai_insights_section(data_loader, analyzer, start_date, end_date)


if __name__ == "__main__":
    main()
