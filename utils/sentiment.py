"""
Sentiment Analysis Module

This module provides sentiment analysis functionality using
Hugging Face transformers for the E-Commerce application.
Includes a fallback rule-based analyzer for restricted environments.
"""

import re
import logging
from typing import Any
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)


class RuleBasedSentimentAnalyzer:
    """
    Fallback rule-based sentiment analyzer using keyword matching.
    Used when transformer models are not available.
    """

    POSITIVE_WORDS = {
        'love', 'amazing', 'excellent', 'fantastic', 'great', 'wonderful',
        'perfect', 'best', 'awesome', 'outstanding', 'superb', 'brilliant',
        'happy', 'pleased', 'satisfied', 'recommend', 'incredible', 'impressive',
        'quality', 'worth', 'easy', 'fast', 'beautiful', 'comfortable',
        'reliable', 'durable', 'premium', 'exceeded', 'expectations', 'flawless',
        'intuitive', 'sleek', 'modern', 'genuine', 'fair', 'loyal', 'saved',
        'quickly', 'super', 'friendly', 'definitely', 'highly', 'five', 'stars',
        'value', 'money', 'works', 'perfectly', 'gift', 'exactly', 'better',
    }

    NEGATIVE_WORDS = {
        'terrible', 'awful', 'horrible', 'worst', 'bad', 'poor', 'disappointed',
        'waste', 'broken', 'broke', 'defective', 'cheap', 'flimsy', 'useless', 'failed',
        'frustrating', 'annoying', 'damaged', 'missing', 'wrong', 'overpriced',
        'slow', 'complicated', 'confusing', 'uncomfortable', 'unreliable',
        'returning', 'refund', 'regret', 'stopped', 'hate', 'dislike',
        'unhappy', 'dissatisfied', 'avoid', 'scam', 'fake', 'lacking',
        'noise', 'weird', 'smaller', 'mediocre', 'unusable', 'junk', 'garbage',
        'disappointing', 'problem', 'problems', 'issue', 'issues', 'error',
    }

    INTENSIFIERS = {
        'very', 'really', 'extremely', 'absolutely', 'totally', 'completely',
        'incredibly', 'highly', 'super', 'so', 'quite', 'definitely',
    }

    NEGATIONS = {'not', "n't", 'no', 'never', 'neither', 'nobody', 'nothing', 'nowhere'}

    def analyze(self, text: str) -> dict[str, Any]:
        """
        Analyze sentiment using rule-based approach.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with 'label' and 'score' keys
        """
        if not text or not text.strip():
            return {'label': 'NEUTRAL', 'score': 0.5}

        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)

        positive_score = 0
        negative_score = 0
        intensifier_active = False
        negation_active = False

        for i, word in enumerate(words):
            # Check for intensifiers
            if word in self.INTENSIFIERS:
                intensifier_active = True
                continue

            # Check for negations
            if word in self.NEGATIONS or word.endswith("n't"):
                negation_active = True
                continue

            # Calculate scores
            multiplier = 1.5 if intensifier_active else 1.0

            if word in self.POSITIVE_WORDS:
                if negation_active:
                    negative_score += multiplier
                else:
                    positive_score += multiplier
            elif word in self.NEGATIVE_WORDS:
                if negation_active:
                    positive_score += multiplier
                else:
                    negative_score += multiplier

            # Reset modifiers
            intensifier_active = False
            negation_active = False

        # Calculate final sentiment
        total = positive_score + negative_score
        if total == 0:
            return {'label': 'NEUTRAL', 'score': 0.5}

        if positive_score > negative_score:
            confidence = 0.5 + (positive_score - negative_score) / (2 * total)
            confidence = min(0.99, max(0.5, confidence))
            return {'label': 'POSITIVE', 'score': confidence}
        elif negative_score > positive_score:
            confidence = 0.5 + (negative_score - positive_score) / (2 * total)
            confidence = min(0.99, max(0.5, confidence))
            return {'label': 'NEGATIVE', 'score': confidence}
        else:
            return {'label': 'POSITIVE', 'score': 0.55}  # Slight positive bias for ties


class SentimentAnalyzer:
    """Handles sentiment analysis using transformer models with fallback."""

    MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

    def __init__(self):
        """Initialize the SentimentAnalyzer."""
        self._pipeline = None
        self._fallback = RuleBasedSentimentAnalyzer()
        self._use_fallback = False

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def _load_model():
        """
        Load the sentiment analysis pipeline.

        Returns:
            Hugging Face pipeline for sentiment analysis, or None if unavailable
        """
        try:
            from transformers import pipeline
            return pipeline(
                "sentiment-analysis",
                model=SentimentAnalyzer.MODEL_NAME,
                device=-1  # Use CPU for compatibility
            )
        except Exception as e:
            logger.warning(f"Could not load transformer model: {e}")
            return None

    @property
    def pipeline(self):
        """Get or create the sentiment analysis pipeline."""
        if self._pipeline is None and not self._use_fallback:
            self._pipeline = self._load_model()
            if self._pipeline is None:
                self._use_fallback = True
                logger.info("Using rule-based sentiment analyzer as fallback")
        return self._pipeline

    def analyze_single(self, text: str) -> dict[str, Any]:
        """
        Analyze sentiment of a single text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with 'label' and 'score' keys
        """
        if not text or not text.strip():
            return {'label': 'NEUTRAL', 'score': 0.5}

        # Use fallback if transformer is not available
        if self._use_fallback or self.pipeline is None:
            return self._fallback.analyze(text)

        # Truncate long texts to avoid model limits
        truncated_text = text[:512]

        try:
            result = self.pipeline(truncated_text)[0]
            return {
                'label': result['label'],
                'score': result['score']
            }
        except Exception as e:
            logger.warning(f"Transformer failed, using fallback: {e}")
            self._use_fallback = True
            return self._fallback.analyze(text)

    @st.cache_data(show_spinner=False)
    def analyze_batch(_self, texts: tuple[str, ...]) -> list[dict[str, Any]]:
        """
        Analyze sentiment of multiple texts efficiently.

        Args:
            texts: Tuple of texts to analyze (tuple for caching)

        Returns:
            List of dictionaries with 'label' and 'score' keys
        """
        if not texts:
            return []

        # Truncate texts and handle empty strings
        processed_texts = []
        valid_indices = []

        for i, text in enumerate(texts):
            if text and text.strip():
                processed_texts.append(text[:512])
                valid_indices.append(i)

        # Initialize results with neutral values
        results = [{'label': 'NEUTRAL', 'score': 0.5} for _ in texts]

        if not processed_texts:
            return results

        # Check if we should use fallback
        if _self._use_fallback or _self.pipeline is None:
            for i, text in enumerate(processed_texts):
                original_idx = valid_indices[i]
                results[original_idx] = _self._fallback.analyze(text)
            return results

        try:
            # Batch process in chunks to avoid memory issues
            chunk_size = 32
            for i in range(0, len(processed_texts), chunk_size):
                chunk = processed_texts[i:i + chunk_size]
                chunk_results = _self.pipeline(chunk)

                for j, result in enumerate(chunk_results):
                    original_idx = valid_indices[i + j]
                    results[original_idx] = {
                        'label': result['label'],
                        'score': result['score']
                    }
        except Exception as e:
            logger.warning(f"Batch processing failed, using fallback: {e}")
            _self._use_fallback = True
            for i, text in enumerate(processed_texts):
                original_idx = valid_indices[i]
                results[original_idx] = _self._fallback.analyze(text)

        return results

    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Add sentiment analysis results to a DataFrame.

        Args:
            df: DataFrame with text data
            text_column: Name of the column containing text

        Returns:
            DataFrame with added 'sentiment_label' and 'sentiment_score' columns
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")

        df = df.copy()

        # Convert to tuple for caching
        texts = tuple(df[text_column].fillna('').astype(str).tolist())

        # Analyze with loading indicator
        results = self.analyze_batch(texts)

        # Add results to DataFrame
        df['sentiment_label'] = [r['label'] for r in results]
        df['sentiment_score'] = [r['score'] for r in results]

        return df

    @property
    def is_using_fallback(self) -> bool:
        """Check if the analyzer is using the fallback method."""
        return self._use_fallback


def get_sentiment_summary(df: pd.DataFrame) -> dict[str, Any]:
    """
    Calculate sentiment summary statistics from analyzed DataFrame.

    Args:
        df: DataFrame with 'sentiment_label' and 'sentiment_score' columns

    Returns:
        Dictionary containing sentiment statistics
    """
    if 'sentiment_label' not in df.columns:
        return {}

    total = len(df)
    positive_df = df[df['sentiment_label'] == 'POSITIVE']
    negative_df = df[df['sentiment_label'] == 'NEGATIVE']
    positive = len(positive_df)
    negative = len(negative_df)

    # Calculate average confidence scores for each sentiment
    positive_avg_conf = positive_df['sentiment_score'].mean() if positive > 0 and 'sentiment_score' in df.columns else 0
    negative_avg_conf = negative_df['sentiment_score'].mean() if negative > 0 and 'sentiment_score' in df.columns else 0
    avg_confidence = df['sentiment_score'].mean() if 'sentiment_score' in df.columns and total > 0 else 0

    return {
        'total': total,
        'positive_count': positive,
        'negative_count': negative,
        'positive_pct': (positive / total * 100) if total > 0 else 0,
        'negative_pct': (negative / total * 100) if total > 0 else 0,
        'avg_confidence': avg_confidence,
        'positive_avg_confidence': positive_avg_conf,
        'negative_avg_confidence': negative_avg_conf,
        # Additional fields for display
        'positive_avg_conf': positive_avg_conf,
        'negative_avg_conf': negative_avg_conf,
    }


def format_sentiment_label(label: str) -> str:
    """
    Format sentiment label for display.

    Args:
        label: Raw sentiment label

    Returns:
        Formatted label string
    """
    label_map = {
        'POSITIVE': 'Positive',
        'NEGATIVE': 'Negative',
        'NEUTRAL': 'Neutral'
    }
    return label_map.get(label.upper(), label.title())
