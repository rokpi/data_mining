"""
Data Loading Utilities

This module handles loading and processing data from JSON files
for the E-Commerce Sentiment Analysis application.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any
import pandas as pd
import streamlit as st


class DataLoader:
    """Handles loading and processing of e-commerce data."""

    def __init__(self, data_dir: str | Path = None):
        """
        Initialize the DataLoader.

        Args:
            data_dir: Path to the data directory. Defaults to ./data
        """
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data"
        self.data_dir = Path(data_dir)

    @st.cache_data(ttl=3600)
    def load_products(_self) -> pd.DataFrame:
        """
        Load products data from JSON file and compute metrics from reviews.

        Returns:
            DataFrame containing product data with computed metrics
        """
        filepath = _self.data_dir / "products.json"

        if not filepath.exists():
            raise FileNotFoundError(f"Products data file not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        df = pd.DataFrame(data)

        # Load reviews to compute product metrics
        try:
            reviews = _self.load_reviews()

            # Compute rating and review count from reviews
            review_stats = reviews.groupby('product_id').agg({
                'rating': 'mean',
                'id': 'count'
            }).rename(columns={'id': 'review_count'})

            # Merge with products
            df = df.merge(review_stats, left_on='id', right_index=True, how='left')

            # Fill NaN values (products with no reviews)
            df['rating'] = df['rating'].fillna(0.0)
            df['review_count'] = df['review_count'].fillna(0).astype(int)
        except Exception:
            # If reviews can't be loaded, use defaults
            if 'rating' not in df.columns:
                df['rating'] = 0.0
            if 'review_count' not in df.columns:
                df['review_count'] = 0

        # Parse date_added if present
        if 'date_added' in df.columns:
            df['date_added'] = pd.to_datetime(df['date_added'])

        return df

    @st.cache_data(ttl=3600)
    def load_reviews(_self) -> pd.DataFrame:
        """
        Load reviews data from JSON file.

        Returns:
            DataFrame containing review data
        """
        filepath = _self.data_dir / "reviews.json"

        if not filepath.exists():
            raise FileNotFoundError(f"Reviews data file not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        df = pd.DataFrame(data)

        # Parse date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['month'] = df['date'].dt.month
            df['month_name'] = df['date'].dt.strftime('%B')
            df['year'] = df['date'].dt.year

        return df

    @st.cache_data(ttl=3600)
    def load_testimonials(_self) -> pd.DataFrame:
        """
        Load testimonials data from JSON file.

        Returns:
            DataFrame containing testimonial data with default values for missing fields
        """
        filepath = _self.data_dir / "testimonials.json"

        if not filepath.exists():
            raise FileNotFoundError(f"Testimonials data file not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        df = pd.DataFrame(data)

        # Parse date column if present
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        # Mark high-rated testimonials as featured
        if 'featured' not in df.columns:
            df['featured'] = df['rating'] >= 5

        return df

    def get_reviews_by_month(self, month: int, year: int = 2023) -> pd.DataFrame:
        """
        Get reviews filtered by specific month.

        Args:
            month: Month number (1-12)
            year: Year (default 2023)

        Returns:
            DataFrame containing filtered reviews
        """
        reviews = self.load_reviews()
        mask = (reviews['month'] == month) & (reviews['year'] == year)
        return reviews[mask].copy()

    def get_monthly_review_counts(self, year: int = 2023) -> dict[str, int]:
        """
        Get review counts by month.

        Args:
            year: Year to filter by

        Returns:
            Dictionary mapping month names to review counts
        """
        reviews = self.load_reviews()
        filtered = reviews[reviews['year'] == year]
        counts = filtered.groupby('month_name').size().to_dict()

        # Ensure all months are present
        month_order = [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]
        return {month: counts.get(month, 0) for month in month_order}

    def get_available_years(self) -> list[int]:
        """
        Get list of years that have data from reviews.

        Returns:
            Sorted list of years with available data
        """
        years = set()

        # Get years from reviews only
        try:
            reviews = self.load_reviews()
            if 'date' in reviews.columns and not reviews.empty:
                review_years = reviews['date'].dt.year.unique()
                years.update(review_years)
        except Exception:
            pass

        # Return sorted list, or default to current year if no data found
        if years:
            return sorted(list(years))
        else:
            from datetime import datetime
            return [datetime.now().year]

    def get_data_statistics(self) -> dict[str, Any]:
        """
        Get overall statistics for all data.

        Returns:
            Dictionary containing data statistics
        """
        products = self.load_products()
        reviews = self.load_reviews()
        testimonials = self.load_testimonials()

        stats = {
            'total_products': len(products),
            'total_reviews': len(reviews),
            'total_testimonials': len(testimonials),
            'categories': products['category'].nunique() if 'category' in products.columns else 0,
            'avg_product_rating': products['rating'].mean() if 'rating' in products.columns else 0,
            'avg_review_rating': reviews['rating'].mean() if 'rating' in reviews.columns else 0,
            'date_range': {
                'start': reviews['date'].min().strftime('%Y-%m-%d') if not reviews.empty else None,
                'end': reviews['date'].max().strftime('%Y-%m-%d') if not reviews.empty else None,
            }
        }
        return stats


def validate_data_files(data_dir: str | Path = None) -> dict[str, bool]:
    """
    Validate that all required data files exist.

    Args:
        data_dir: Path to data directory

    Returns:
        Dictionary mapping filenames to existence status
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data"
    data_dir = Path(data_dir)

    required_files = ['products.json', 'reviews.json', 'testimonials.json']

    return {
        filename: (data_dir / filename).exists()
        for filename in required_files
    }
