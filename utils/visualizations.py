"""
Visualization Module

This module provides chart creation functions using Plotly
for the E-Commerce Sentiment Analysis application.
"""

from typing import Any, Optional
from collections import Counter
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Try to import wordcloud, fallback gracefully if not available
try:
    from wordcloud import WordCloud
    import io
    import base64
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False


# Color schemes
COLORS = {
    'positive': '#10B981',  # Green
    'negative': '#EF4444',  # Red
    'neutral': '#6B7280',   # Gray
    'primary': '#3B82F6',   # Blue
    'secondary': '#8B5CF6', # Purple
    'background': '#1F2937',
    'card': '#374151',
    'text': '#F9FAFB',
}


def create_sentiment_chart(
    df: pd.DataFrame,
    title: str = "Sentiment Distribution"
) -> go.Figure:
    """
    Create an interactive bar chart showing sentiment distribution with confidence scores.

    Args:
        df: DataFrame with 'sentiment_label' column
        title: Chart title

    Returns:
        Plotly Figure object
    """
    if 'sentiment_label' not in df.columns:
        return go.Figure()

    # Count sentiments
    sentiment_counts = df['sentiment_label'].value_counts()

    # Calculate percentages and average confidence
    total = len(df)
    data = []

    for label in ['POSITIVE', 'NEGATIVE']:
        count = sentiment_counts.get(label, 0)
        pct = (count / total * 100) if total > 0 else 0
        avg_conf = df[df['sentiment_label'] == label]['sentiment_score'].mean() if count > 0 else 0

        data.append({
            'Sentiment': label.title(),
            'Count': count,
            'Percentage': pct,
            'Avg Confidence': avg_conf
        })

    chart_df = pd.DataFrame(data)

    # Create figure
    fig = go.Figure()

    # Main bars with confidence score displayed prominently
    fig.add_trace(go.Bar(
        x=chart_df['Sentiment'],
        y=chart_df['Count'],
        marker_color=[COLORS['positive'], COLORS['negative']],
        text=[
            f"<b>{row['Count']}</b><br>"
            f"({row['Percentage']:.1f}%)<br>"
            f"<span style='font-size:12px'>Avg Confidence: {row['Avg Confidence']:.0%}</span>"
            for _, row in chart_df.iterrows()
        ],
        textposition='outside',
        textfont=dict(size=14, color=COLORS['text']),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Count: %{y}<br>"
            "Percentage: %{customdata[0]:.1f}%<br>"
            "Avg Confidence: %{customdata[1]:.0%}<br>"
            "<extra></extra>"
        ),
        customdata=chart_df[['Percentage', 'Avg Confidence']].values
    ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, color=COLORS['text']),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title="",
            tickfont=dict(size=14, color=COLORS['text']),
            gridcolor='rgba(255,255,255,0.1)'
        ),
        yaxis=dict(
            title="Number of Reviews",
            titlefont=dict(size=14, color=COLORS['text']),
            tickfont=dict(size=12, color=COLORS['text']),
            gridcolor='rgba(255,255,255,0.1)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=100, b=60, l=60, r=40),
        height=450,
        showlegend=False,
    )

    return fig


def create_confidence_histogram(
    df: pd.DataFrame,
    title: str = "Confidence Score Distribution"
) -> go.Figure:
    """
    Create a histogram of confidence scores by sentiment.

    Args:
        df: DataFrame with 'sentiment_score' and 'sentiment_label' columns
        title: Chart title

    Returns:
        Plotly Figure object
    """
    if 'sentiment_score' not in df.columns or 'sentiment_label' not in df.columns:
        return go.Figure()

    fig = go.Figure()

    for label, color in [('POSITIVE', COLORS['positive']), ('NEGATIVE', COLORS['negative'])]:
        mask = df['sentiment_label'] == label
        if mask.any():
            fig.add_trace(go.Histogram(
                x=df[mask]['sentiment_score'],
                name=label.title(),
                marker_color=color,
                opacity=0.7,
                nbinsx=20,
            ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, color=COLORS['text']),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title="Confidence Score",
            titlefont=dict(size=14, color=COLORS['text']),
            tickfont=dict(size=12, color=COLORS['text']),
            range=[0.5, 1.0],
            gridcolor='rgba(255,255,255,0.1)'
        ),
        yaxis=dict(
            title="Frequency",
            titlefont=dict(size=14, color=COLORS['text']),
            tickfont=dict(size=12, color=COLORS['text']),
            gridcolor='rgba(255,255,255,0.1)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        barmode='overlay',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            font=dict(color=COLORS['text'])
        ),
        margin=dict(t=80, b=60, l=60, r=40),
        height=350,
    )

    return fig


def create_trend_chart(
    monthly_data: dict[str, dict[str, Any]],
    title: str = "Sentiment Trend (2023)"
) -> go.Figure:
    """
    Create a line chart showing sentiment trends across months.

    Args:
        monthly_data: Dictionary with month names as keys and sentiment data as values
        title: Chart title

    Returns:
        Plotly Figure object
    """
    months = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]

    positive_pcts = []
    negative_pcts = []
    totals = []

    for month in months:
        data = monthly_data.get(month, {})
        positive_pcts.append(data.get('positive_pct', 0))
        negative_pcts.append(data.get('negative_pct', 0))
        totals.append(data.get('total', 0))

    fig = go.Figure()

    # Positive trend line
    fig.add_trace(go.Scatter(
        x=months,
        y=positive_pcts,
        mode='lines+markers',
        name='Positive %',
        line=dict(color=COLORS['positive'], width=3),
        marker=dict(size=8),
        hovertemplate="<b>%{x}</b><br>Positive: %{y:.1f}%<br>Reviews: %{customdata}<extra></extra>",
        customdata=totals
    ))

    # Negative trend line
    fig.add_trace(go.Scatter(
        x=months,
        y=negative_pcts,
        mode='lines+markers',
        name='Negative %',
        line=dict(color=COLORS['negative'], width=3),
        marker=dict(size=8),
        hovertemplate="<b>%{x}</b><br>Negative: %{y:.1f}%<br>Reviews: %{customdata}<extra></extra>",
        customdata=totals
    ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, color=COLORS['text']),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title="",
            tickfont=dict(size=11, color=COLORS['text']),
            tickangle=45,
            gridcolor='rgba(255,255,255,0.1)'
        ),
        yaxis=dict(
            title="Percentage",
            titlefont=dict(size=14, color=COLORS['text']),
            tickfont=dict(size=12, color=COLORS['text']),
            ticksuffix='%',
            gridcolor='rgba(255,255,255,0.1)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            font=dict(color=COLORS['text'])
        ),
        margin=dict(t=80, b=100, l=60, r=40),
        height=400,
        hovermode='x unified'
    )

    return fig


def create_rating_distribution_chart(
    df: pd.DataFrame,
    title: str = "Rating Distribution"
) -> go.Figure:
    """
    Create a bar chart showing rating distribution.

    Args:
        df: DataFrame with 'rating' column
        title: Chart title

    Returns:
        Plotly Figure object
    """
    if 'rating' not in df.columns:
        return go.Figure()

    rating_counts = df['rating'].value_counts().sort_index()

    # Define colors for ratings (1-5 stars)
    rating_colors = {
        1: '#EF4444',  # Red
        2: '#F97316',  # Orange
        3: '#EAB308',  # Yellow
        4: '#84CC16',  # Light green
        5: '#10B981',  # Green
    }

    colors = [rating_colors.get(r, COLORS['primary']) for r in rating_counts.index]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=[f"{r} {'Stars' if r != 1 else 'Star'}" for r in rating_counts.index],
        y=rating_counts.values,
        marker_color=colors,
        text=rating_counts.values,
        textposition='outside',
        textfont=dict(size=12, color=COLORS['text']),
        hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color=COLORS['text']),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title="",
            tickfont=dict(size=12, color=COLORS['text']),
            gridcolor='rgba(255,255,255,0.1)'
        ),
        yaxis=dict(
            title="Number of Reviews",
            titlefont=dict(size=12, color=COLORS['text']),
            tickfont=dict(size=10, color=COLORS['text']),
            gridcolor='rgba(255,255,255,0.1)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=60, b=60, l=60, r=40),
        height=300,
        showlegend=False,
    )

    return fig


def create_category_chart(
    df: pd.DataFrame,
    title: str = "Reviews by Category"
) -> go.Figure:
    """
    Create a horizontal bar chart showing reviews by category.

    Args:
        df: DataFrame with 'category' column
        title: Chart title

    Returns:
        Plotly Figure object
    """
    if 'category' not in df.columns:
        return go.Figure()

    category_counts = df['category'].value_counts()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=category_counts.index,
        x=category_counts.values,
        orientation='h',
        marker_color=COLORS['primary'],
        text=category_counts.values,
        textposition='outside',
        textfont=dict(size=12, color=COLORS['text']),
        hovertemplate="<b>%{y}</b><br>Reviews: %{x}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color=COLORS['text']),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title="Number of Reviews",
            titlefont=dict(size=12, color=COLORS['text']),
            tickfont=dict(size=10, color=COLORS['text']),
            gridcolor='rgba(255,255,255,0.1)'
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=11, color=COLORS['text']),
            gridcolor='rgba(255,255,255,0.1)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=60, b=60, l=150, r=60),
        height=300,
        showlegend=False,
    )

    return fig


def create_sentiment_by_rating_chart(
    df: pd.DataFrame,
    title: str = "Sentiment vs. Star Rating"
) -> go.Figure:
    """
    Create a grouped bar chart showing sentiment distribution by rating.

    Args:
        df: DataFrame with 'rating' and 'sentiment_label' columns
        title: Chart title

    Returns:
        Plotly Figure object
    """
    if 'rating' not in df.columns or 'sentiment_label' not in df.columns:
        return go.Figure()

    # Create cross-tabulation
    crosstab = pd.crosstab(df['rating'], df['sentiment_label'])

    fig = go.Figure()

    if 'POSITIVE' in crosstab.columns:
        fig.add_trace(go.Bar(
            x=[f"{r} {'Stars' if r != 1 else 'Star'}" for r in crosstab.index],
            y=crosstab['POSITIVE'],
            name='Positive',
            marker_color=COLORS['positive'],
        ))

    if 'NEGATIVE' in crosstab.columns:
        fig.add_trace(go.Bar(
            x=[f"{r} {'Stars' if r != 1 else 'Star'}" for r in crosstab.index],
            y=crosstab['NEGATIVE'],
            name='Negative',
            marker_color=COLORS['negative'],
        ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color=COLORS['text']),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title="Star Rating",
            titlefont=dict(size=12, color=COLORS['text']),
            tickfont=dict(size=10, color=COLORS['text']),
            gridcolor='rgba(255,255,255,0.1)'
        ),
        yaxis=dict(
            title="Count",
            titlefont=dict(size=12, color=COLORS['text']),
            tickfont=dict(size=10, color=COLORS['text']),
            gridcolor='rgba(255,255,255,0.1)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        barmode='group',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            font=dict(color=COLORS['text'])
        ),
        margin=dict(t=80, b=60, l=60, r=40),
        height=350,
    )

    return fig


def create_word_cloud(
    texts: list[str],
    sentiment: str = 'all',
    max_words: int = 100
) -> Optional[str]:
    """
    Create a word cloud from texts and return as base64 encoded image.

    Args:
        texts: List of text strings
        sentiment: Filter by sentiment ('positive', 'negative', or 'all')
        max_words: Maximum number of words to include

    Returns:
        Base64 encoded PNG image string or None if wordcloud not available
    """
    if not WORDCLOUD_AVAILABLE:
        return None

    if not texts:
        return None

    # Combine all texts
    combined_text = ' '.join(texts)

    # Remove common stop words and clean text
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'this', 'that', 'these',
        'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which',
        'who', 'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both',
        'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
        'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'my', 'your'
    }

    # Set colors based on sentiment
    if sentiment == 'positive':
        colormap = 'Greens'
        background_color = '#1F2937'
    elif sentiment == 'negative':
        colormap = 'Reds'
        background_color = '#1F2937'
    else:
        colormap = 'Blues'
        background_color = '#1F2937'

    try:
        wordcloud = WordCloud(
            width=800,
            height=400,
            max_words=max_words,
            background_color=background_color,
            colormap=colormap,
            stopwords=stop_words,
            min_font_size=10,
            max_font_size=100,
            prefer_horizontal=0.7,
        ).generate(combined_text)

        # Convert to image
        img_buffer = io.BytesIO()
        wordcloud.to_image().save(img_buffer, format='PNG')
        img_buffer.seek(0)

        # Encode to base64
        img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"

    except Exception:
        return None


def create_comparison_chart(
    data1: dict[str, Any],
    data2: dict[str, Any],
    label1: str = "Period 1",
    label2: str = "Period 2"
) -> go.Figure:
    """
    Create a side-by-side comparison chart for two periods.

    Args:
        data1: Sentiment data for period 1
        data2: Sentiment data for period 2
        label1: Label for period 1
        label2: Label for period 2

    Returns:
        Plotly Figure object
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(label1, label2),
        specs=[[{'type': 'domain'}, {'type': 'domain'}]]
    )

    # Period 1 pie chart
    fig.add_trace(go.Pie(
        labels=['Positive', 'Negative'],
        values=[data1.get('positive_count', 0), data1.get('negative_count', 0)],
        marker_colors=[COLORS['positive'], COLORS['negative']],
        textinfo='percent+label',
        textfont=dict(size=12, color=COLORS['text']),
        hole=0.4,
        name=label1
    ), row=1, col=1)

    # Period 2 pie chart
    fig.add_trace(go.Pie(
        labels=['Positive', 'Negative'],
        values=[data2.get('positive_count', 0), data2.get('negative_count', 0)],
        marker_colors=[COLORS['positive'], COLORS['negative']],
        textinfo='percent+label',
        textfont=dict(size=12, color=COLORS['text']),
        hole=0.4,
        name=label2
    ), row=1, col=2)

    fig.update_layout(
        title=dict(
            text="Sentiment Comparison",
            font=dict(size=20, color=COLORS['text']),
            x=0.5,
            xanchor='center'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS['text']),
        height=400,
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.1,
            xanchor='center',
            x=0.5,
            font=dict(color=COLORS['text'])
        )
    )

    return fig


def create_confidence_trend_chart(
    monthly_data: dict[str, dict[str, Any]],
    title: str = "Confidence Score Trend (2023)"
) -> go.Figure:
    """
    Create a line chart showing confidence score trends across months.

    Args:
        monthly_data: Dictionary with month names as keys and sentiment data as values
        title: Chart title

    Returns:
        Plotly Figure object
    """
    months = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]

    positive_conf = []
    negative_conf = []
    avg_conf = []

    for month in months:
        data = monthly_data.get(month, {})
        positive_conf.append(data.get('positive_avg_conf', 0))
        negative_conf.append(data.get('negative_avg_conf', 0))
        avg_conf.append(data.get('avg_confidence', 0))

    fig = go.Figure()

    # Average confidence line
    fig.add_trace(go.Scatter(
        x=months,
        y=avg_conf,
        mode='lines+markers',
        name='Average Confidence',
        line=dict(color=COLORS['primary'], width=3),
        marker=dict(size=8),
        hovertemplate="<b>%{x}</b><br>Avg Confidence: %{y:.1%}<extra></extra>"
    ))

    # Positive confidence line
    fig.add_trace(go.Scatter(
        x=months,
        y=positive_conf,
        mode='lines+markers',
        name='Positive Confidence',
        line=dict(color=COLORS['positive'], width=2, dash='dash'),
        marker=dict(size=6),
        hovertemplate="<b>%{x}</b><br>Positive Conf: %{y:.1%}<extra></extra>"
    ))

    # Negative confidence line
    fig.add_trace(go.Scatter(
        x=months,
        y=negative_conf,
        mode='lines+markers',
        name='Negative Confidence',
        line=dict(color=COLORS['negative'], width=2, dash='dash'),
        marker=dict(size=6),
        hovertemplate="<b>%{x}</b><br>Negative Conf: %{y:.1%}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, color=COLORS['text']),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title="",
            tickfont=dict(size=11, color=COLORS['text']),
            tickangle=45,
            gridcolor='rgba(255,255,255,0.1)'
        ),
        yaxis=dict(
            title="Confidence Score",
            titlefont=dict(size=14, color=COLORS['text']),
            tickfont=dict(size=12, color=COLORS['text']),
            tickformat='.0%',
            gridcolor='rgba(255,255,255,0.1)',
            range=[0.5, 1.0]
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            font=dict(color=COLORS['text'])
        ),
        margin=dict(t=80, b=100, l=60, r=40),
        height=400,
        hovermode='x unified'
    )

    return fig


def create_product_performance_chart(
    df: pd.DataFrame,
    top_n: int = 10
) -> go.Figure:
    """
    Create a horizontal bar chart showing product performance by sentiment.

    Args:
        df: DataFrame with product_name and sentiment_score columns
        top_n: Number of top products to show

    Returns:
        Plotly Figure object
    """
    if 'product_name' not in df.columns or 'sentiment_score' not in df.columns:
        return go.Figure()

    # Calculate average sentiment score per product
    product_scores = df.groupby('product_name').agg({
        'sentiment_score': 'mean',
        'sentiment_label': lambda x: (x == 'POSITIVE').mean() * 100
    }).reset_index()

    product_scores.columns = ['Product', 'Avg Confidence', 'Positive %']

    # Sort by positive percentage and take top N
    product_scores = product_scores.sort_values('Positive %', ascending=True).tail(top_n)

    # Determine colors based on positive percentage
    colors = [
        COLORS['positive'] if pct >= 60 else COLORS['negative'] if pct < 40 else COLORS['neutral']
        for pct in product_scores['Positive %']
    ]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=product_scores['Product'],
        x=product_scores['Positive %'],
        orientation='h',
        marker_color=colors,
        text=[f"{pct:.0f}% positive" for pct in product_scores['Positive %']],
        textposition='outside',
        textfont=dict(size=11, color=COLORS['text']),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Positive: %{x:.1f}%<br>"
            "Avg Confidence: %{customdata:.1%}<extra></extra>"
        ),
        customdata=product_scores['Avg Confidence']
    ))

    fig.update_layout(
        title=dict(
            text=f"Top {top_n} Products by Sentiment",
            font=dict(size=18, color=COLORS['text']),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title="Positive Sentiment (%)",
            titlefont=dict(size=12, color=COLORS['text']),
            tickfont=dict(size=10, color=COLORS['text']),
            ticksuffix='%',
            gridcolor='rgba(255,255,255,0.1)',
            range=[0, 100]
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=10, color=COLORS['text']),
            gridcolor='rgba(255,255,255,0.1)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=60, b=60, l=200, r=80),
        height=400,
        showlegend=False,
    )

    return fig


def extract_common_phrases(
    texts: list[str],
    min_word_length: int = 3,
    top_n: int = 20
) -> list[tuple[str, int]]:
    """
    Extract most common phrases/words from texts.

    Args:
        texts: List of text strings
        min_word_length: Minimum word length to include
        top_n: Number of top phrases to return

    Returns:
        List of (phrase, count) tuples
    """
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'this', 'that', 'these',
        'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which',
        'who', 'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both',
        'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
        'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'my', 'your',
        'its', 'ive', 'ita'
    }

    word_counts = Counter()

    for text in texts:
        # Clean and tokenize
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        for word in words:
            if len(word) >= min_word_length and word not in stop_words:
                word_counts[word] += 1

    return word_counts.most_common(top_n)
