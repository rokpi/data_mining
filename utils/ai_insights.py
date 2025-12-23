"""
AI Insights Module

This module provides AI-powered analysis and insights generation
for products and testimonials using NLP techniques.
"""

import re
from collections import Counter
from typing import Any, Optional
import pandas as pd

# Try to import transformers for advanced NLP
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class AIInsightsAnalyzer:
    """
    AI-powered analyzer for generating insights from reviews and testimonials.
    Uses a combination of NLP techniques and rule-based analysis.
    """

    # Positive sentiment keywords with weights
    POSITIVE_KEYWORDS = {
        'love': 3, 'excellent': 3, 'amazing': 3, 'perfect': 3, 'outstanding': 3,
        'fantastic': 3, 'wonderful': 3, 'incredible': 3, 'awesome': 3,
        'great': 2, 'good': 1, 'nice': 1, 'best': 3, 'recommend': 2,
        'quality': 2, 'fast': 2, 'easy': 2, 'beautiful': 2, 'reliable': 2,
        'impressed': 2, 'satisfied': 2, 'happy': 2, 'delighted': 3,
        'value': 2, 'worth': 2, 'durable': 2, 'comfortable': 2,
        'helpful': 2, 'efficient': 2, 'responsive': 2, 'friendly': 2,
    }

    # Negative sentiment keywords with weights
    NEGATIVE_KEYWORDS = {
        'terrible': 3, 'awful': 3, 'horrible': 3, 'worst': 3, 'hate': 3,
        'disappointed': 2, 'disappointing': 2, 'frustrating': 2, 'frustrated': 2,
        'poor': 2, 'bad': 2, 'broken': 3, 'defective': 3, 'useless': 3,
        'waste': 2, 'expensive': 1, 'overpriced': 2, 'cheap': 1,
        'slow': 2, 'difficult': 1, 'confusing': 1, 'complicated': 1,
        'unhelpful': 2, 'rude': 2, 'damaged': 2, 'missing': 2,
        'return': 1, 'refund': 1, 'problem': 2, 'issue': 1, 'failed': 2,
    }

    # Topic categories for classification
    TOPIC_KEYWORDS = {
        'quality': ['quality', 'build', 'material', 'durable', 'sturdy', 'solid', 'well-made', 'craftsmanship'],
        'price': ['price', 'value', 'money', 'expensive', 'cheap', 'affordable', 'cost', 'worth', 'deal'],
        'shipping': ['shipping', 'delivery', 'arrived', 'fast', 'quick', 'slow', 'packaging', 'box'],
        'customer_service': ['service', 'support', 'help', 'response', 'customer', 'staff', 'team', 'contact'],
        'ease_of_use': ['easy', 'simple', 'intuitive', 'user-friendly', 'complicated', 'confusing', 'setup'],
        'design': ['design', 'look', 'beautiful', 'ugly', 'style', 'appearance', 'aesthetic', 'color'],
        'performance': ['performance', 'works', 'function', 'effective', 'efficient', 'powerful', 'speed'],
        'durability': ['durable', 'lasting', 'broke', 'broken', 'sturdy', 'fragile', 'reliable'],
    }

    def __init__(self):
        """Initialize the AI insights analyzer."""
        self._summarizer = None

    def _get_summarizer(self):
        """Lazy load summarization pipeline."""
        if self._summarizer is None and TRANSFORMERS_AVAILABLE:
            try:
                self._summarizer = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    max_length=100,
                    min_length=30
                )
            except Exception:
                pass
        return self._summarizer

    def extract_keywords(
        self,
        texts: list[str],
        sentiment_filter: Optional[str] = None,
        top_n: int = 10
    ) -> list[tuple[str, int]]:
        """
        Extract most frequent meaningful keywords from texts.

        Args:
            texts: List of text strings
            sentiment_filter: 'positive', 'negative', or None for all
            top_n: Number of keywords to return

        Returns:
            List of (keyword, count) tuples
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
            'its', 'ive', 'product', 'item', 'one', 'also', 'got', 'get', 'would',
            've', 's', 't', 'really', 'much', 'even', 'well'
        }

        word_counts = Counter()

        for text in texts:
            words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
            for word in words:
                if len(word) >= 3 and word not in stop_words:
                    word_counts[word] += 1

        return word_counts.most_common(top_n)

    def identify_topics(self, texts: list[str]) -> dict[str, int]:
        """
        Identify main topics discussed in the texts.

        Args:
            texts: List of text strings

        Returns:
            Dictionary of topic -> mention count
        """
        topic_counts = {topic: 0 for topic in self.TOPIC_KEYWORDS}

        for text in texts:
            text_lower = text.lower()
            for topic, keywords in self.TOPIC_KEYWORDS.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        topic_counts[topic] += 1
                        break  # Count each topic only once per text

        return topic_counts

    def analyze_product_reviews(
        self,
        product_name: str,
        reviews_df: pd.DataFrame
    ) -> dict[str, Any]:
        """
        Analyze reviews for a specific product and generate actionable insights.

        Args:
            product_name: Name of the product
            reviews_df: DataFrame containing reviews with sentiment analysis

        Returns:
            Dictionary containing performance metrics and action suggestions
        """
        if reviews_df.empty:
            return {
                'product_name': product_name,
                'total_reviews': 0,
                'avg_rating': 0,
                'positive_count': 0,
                'negative_count': 0,
                'action_suggestions': ['No reviews available for analysis'],
                'important_phrases': [],
            }

        # Filter for this product
        product_reviews = reviews_df[reviews_df['product_name'] == product_name].copy()

        # Performance Metrics
        total = len(product_reviews)
        avg_rating = product_reviews['rating'].mean() if 'rating' in product_reviews.columns and not product_reviews.empty else 0
        positive_count = (product_reviews['sentiment_label'] == 'POSITIVE').sum() if not product_reviews.empty else 0
        negative_count = (product_reviews['sentiment_label'] == 'NEGATIVE').sum() if not product_reviews.empty else 0

        # Extract important phrases and generate suggestions only if we have reviews
        if not product_reviews.empty:
            # Extract important phrases from all reviews
            all_texts = product_reviews['text'].tolist()
            important_phrases = self._extract_important_sentences(all_texts, top_n=5)

            # Generate action suggestions
            action_suggestions = self._generate_action_suggestions(
                product_reviews, avg_rating, positive_count, negative_count
            )
        else:
            important_phrases = []
            action_suggestions = [f'No reviews available for {product_name} in the selected period']

        return {
            'product_name': product_name,
            'total_reviews': total,
            'avg_rating': avg_rating,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'action_suggestions': action_suggestions,
            'important_phrases': important_phrases,
        }

    def _extract_important_sentences(
        self,
        texts: list[str],
        top_n: int = 5
    ) -> list[str]:
        """
        Extract the most important and frequently mentioned phrases from reviews.

        Args:
            texts: List of review texts
            top_n: Number of important phrases to return

        Returns:
            List of important phrases
        """
        # Extract keywords
        keywords = self.extract_keywords(texts, top_n=top_n * 2)

        # Find sentences containing these keywords
        important = []
        keyword_words = [kw[0] for kw in keywords[:10]]

        for text in texts:
            sentences = re.split(r'[.!?]+', text)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20 and len(sentence) < 150:
                    sentence_lower = sentence.lower()
                    # Check if sentence contains important keywords
                    if any(kw in sentence_lower for kw in keyword_words):
                        if sentence not in important:
                            important.append(sentence)
                            if len(important) >= top_n:
                                return important

        return important[:top_n]

    def _generate_action_suggestions(
        self,
        reviews_df: pd.DataFrame,
        avg_rating: float,
        positive_count: int,
        negative_count: int
    ) -> list[str]:
        """
        Generate actionable suggestions based on review analysis.

        Args:
            reviews_df: DataFrame with reviews
            avg_rating: Average rating
            positive_count: Number of positive reviews
            negative_count: Number of negative reviews

        Returns:
            List of actionable suggestions
        """
        suggestions = []
        total = len(reviews_df)

        # Get positive and negative reviews
        positive_reviews = reviews_df[reviews_df['sentiment_label'] == 'POSITIVE']
        negative_reviews = reviews_df[reviews_df['sentiment_label'] == 'NEGATIVE']

        positive_texts = positive_reviews['text'].tolist()
        negative_texts = negative_reviews['text'].tolist()

        # Analyze what to improve
        if negative_count > 0:
            # Extract issues from negative reviews
            negative_keywords = self.extract_keywords(negative_texts, top_n=5)
            topics = self.identify_topics(negative_texts)

            # Priority issues
            for word, count in negative_keywords[:3]:
                if word in ['quality', 'broken', 'defective', 'cheap']:
                    suggestions.append(f"⚠️ Improve product quality - '{word}' mentioned {count} times in negative reviews")
                elif word in ['expensive', 'price', 'overpriced']:
                    suggestions.append(f"💰 Review pricing strategy - customers find it too expensive ({count} mentions)")
                elif word in ['shipping', 'delivery', 'slow', 'late']:
                    suggestions.append(f"📦 Speed up delivery - shipping issues mentioned {count} times")
                elif word in ['size', 'sizing', 'fit', 'small', 'large']:
                    suggestions.append(f"📏 Update sizing information - size issues mentioned {count} times")

            # Topic-based suggestions
            sorted_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)
            for topic, count in sorted_topics[:2]:
                if count > 0:
                    if topic == 'quality':
                        suggestions.append(f"🔍 Conduct quality control review - mentioned in {count} reviews")
                    elif topic == 'customer_service':
                        suggestions.append(f"📞 Improve customer service response time - {count} mentions")
                    elif topic == 'durability':
                        suggestions.append(f"🛠️ Address durability concerns - mentioned {count} times")

        # Leverage strengths from positive reviews
        if positive_count > 0:
            positive_keywords = self.extract_keywords(positive_texts, top_n=5)

            for word, count in positive_keywords[:2]:
                if count >= 3:
                    if word in ['love', 'great', 'excellent', 'amazing', 'perfect']:
                        suggestions.append(f"✨ Highlight in marketing: '{word}' mentioned {count} times positively")
                    elif word in ['comfortable', 'quality', 'durable', 'beautiful']:
                        suggestions.append(f"💎 Feature '{word}' in product description - {count} positive mentions")

        # Rating-based suggestions
        if avg_rating < 3.5:
            suggestions.append("🚨 URGENT: Consider product redesign or discontinuation - low rating")
        elif avg_rating < 4.0:
            suggestions.append("📊 Investigate common complaints and address top issues")
        elif avg_rating >= 4.5:
            suggestions.append("⭐ Encourage reviews from satisfied customers to maintain high rating")

        # Conversion suggestions
        positive_pct = (positive_count / total * 100) if total > 0 else 0
        if positive_pct >= 80:
            suggestions.append("📈 Use customer testimonials in advertising - high satisfaction rate")
        elif positive_pct < 50:
            suggestions.append("🔄 Offer satisfaction guarantee or easy returns to build trust")

        # Ensure we have at least some suggestions
        if not suggestions:
            suggestions.append("✅ Maintain current product quality and service standards")
            suggestions.append("📣 Encourage more customer reviews for better insights")

        return suggestions[:6]  # Return top 6 actionable suggestions

    def _generate_praise_points(
        self,
        texts: list[str],
        keywords: list[tuple[str, int]]
    ) -> list[str]:
        """Generate human-readable praise points from positive reviews."""
        praise_points = []

        # Map keywords to praise statements
        praise_templates = {
            'quality': 'High quality materials and construction',
            'love': 'Customers love this product',
            'excellent': 'Excellent overall experience',
            'great': 'Great value for money',
            'fast': 'Fast shipping and delivery',
            'easy': 'Easy to use and set up',
            'perfect': 'Perfect fit for customer needs',
            'recommend': 'Highly recommended by customers',
            'works': 'Works exactly as described',
            'beautiful': 'Beautiful design and appearance',
            'comfortable': 'Very comfortable to use',
            'durable': 'Durable and long-lasting',
            'value': 'Excellent value for the price',
            'shipping': 'Fast and reliable shipping',
            'price': 'Competitive pricing',
            'design': 'Attractive design',
            'service': 'Excellent customer service',
            'amazing': 'Amazing product features',
            'best': 'One of the best in its category',
        }

        for word, count in keywords:
            if word in praise_templates:
                point = f"{praise_templates[word]} (mentioned {count} times)"
                praise_points.append(point)
            elif count >= 3:
                # Generic praise point for frequent keywords
                point = f"Customers frequently mention '{word}' positively ({count} mentions)"
                praise_points.append(point)

        # If no specific points, generate generic ones
        if not praise_points and texts:
            praise_points = [
                "Customers are generally satisfied with the product",
                "Positive reviews highlight good user experience"
            ]

        return praise_points

    def _generate_complaint_points(
        self,
        texts: list[str],
        keywords: list[tuple[str, int]]
    ) -> list[str]:
        """Generate human-readable complaint points from negative reviews."""
        complaint_points = []

        # Map keywords to complaint statements
        complaint_templates = {
            'quality': 'Quality concerns reported by some customers',
            'disappointed': 'Some customers expressed disappointment',
            'broken': 'Product arrived broken or became defective',
            'expensive': 'Some customers find the price too high',
            'price': 'Pricing concerns mentioned',
            'slow': 'Slow shipping or performance issues',
            'difficult': 'Difficulty with setup or usage',
            'return': 'Some customers requested returns',
            'problem': 'Various problems reported',
            'issue': 'Issues with product functionality',
            'damaged': 'Products arrived damaged',
            'poor': 'Poor quality or performance',
            'waste': 'Considered a waste of money by some',
            'confusing': 'Confusing instructions or interface',
            'service': 'Customer service complaints',
            'missing': 'Missing parts or features',
        }

        for word, count in keywords:
            if word in complaint_templates:
                point = f"{complaint_templates[word]} ({count} mentions)"
                complaint_points.append(point)
            elif count >= 2:
                # Generic complaint point for frequent keywords
                point = f"Issue with '{word}' mentioned {count} times"
                complaint_points.append(point)

        # If no specific points, generate generic ones
        if not complaint_points and texts:
            complaint_points = [
                "Some customers had negative experiences",
                "Areas for improvement identified in reviews"
            ]

        return complaint_points

    def _generate_recommendations(
        self,
        positive_pct: float,
        praise_points: list[str],
        complaint_points: list[str],
        topics: dict[str, int]
    ) -> list[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []

        # Based on sentiment ratio
        if positive_pct < 60:
            recommendations.append(
                "Priority: Address common complaints to improve overall satisfaction"
            )

        # Based on topics
        sorted_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)

        for topic, count in sorted_topics[:3]:
            if count > 0:
                if topic == 'price':
                    recommendations.append(
                        "Consider introducing a value pack or discount program to address pricing concerns"
                    )
                elif topic == 'quality':
                    recommendations.append(
                        "Highlight quality certifications and materials in product descriptions"
                    )
                elif topic == 'shipping':
                    recommendations.append(
                        "Review shipping partners and packaging to ensure fast, safe delivery"
                    )
                elif topic == 'customer_service':
                    recommendations.append(
                        "Invest in customer service training and faster response times"
                    )
                elif topic == 'ease_of_use':
                    recommendations.append(
                        "Consider adding video tutorials or improved instructions"
                    )
                elif topic == 'durability':
                    recommendations.append(
                        "Review product durability and consider warranty extensions"
                    )

        # Based on praise points - leverage strengths
        if praise_points:
            recommendations.append(
                "Leverage positive feedback in marketing campaigns - highlight what customers love"
            )

        # General recommendations
        if positive_pct >= 80:
            recommendations.append(
                "Encourage satisfied customers to leave reviews to maintain high ratings"
            )
        elif positive_pct < 40:
            recommendations.append(
                "Conduct direct outreach to dissatisfied customers for detailed feedback"
            )

        return recommendations

    def analyze_testimonials(self, testimonials_df: pd.DataFrame) -> dict[str, Any]:
        """
        Analyze all testimonials and generate actionable insights.

        Args:
            testimonials_df: DataFrame containing testimonials

        Returns:
            Dictionary containing performance metrics and action suggestions
        """
        if testimonials_df.empty:
            return {
                'total_testimonials': 0,
                'avg_rating': 0,
                'positive_count': 0,
                'negative_count': 0,
                'action_suggestions': ['No testimonials available for analysis'],
                'important_phrases': [],
            }

        # Add sentiment analysis if not present
        if 'sentiment_label' not in testimonials_df.columns:
            # Simple sentiment based on rating
            testimonials_df = testimonials_df.copy()
            testimonials_df['sentiment_label'] = testimonials_df['rating'].apply(
                lambda x: 'POSITIVE' if x >= 4 else 'NEGATIVE'
            )

        # Performance Metrics
        total = len(testimonials_df)
        avg_rating = testimonials_df['rating'].mean() if 'rating' in testimonials_df.columns else 0
        positive_count = (testimonials_df['sentiment_label'] == 'POSITIVE').sum()
        negative_count = (testimonials_df['sentiment_label'] == 'NEGATIVE').sum()

        # Extract important phrases from all testimonials
        all_texts = testimonials_df['text'].tolist()
        important_phrases = self._extract_important_sentences(all_texts, top_n=5)

        # Generate action suggestions
        action_suggestions = self._generate_testimonial_action_suggestions(
            testimonials_df, avg_rating, positive_count, negative_count
        )

        return {
            'total_testimonials': total,
            'avg_rating': avg_rating,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'action_suggestions': action_suggestions,
            'important_phrases': important_phrases,
        }

    def _generate_testimonial_action_suggestions(
        self,
        testimonials_df: pd.DataFrame,
        avg_rating: float,
        positive_count: int,
        negative_count: int
    ) -> list[str]:
        """
        Generate actionable suggestions based on testimonial analysis.

        Args:
            testimonials_df: DataFrame with testimonials
            avg_rating: Average rating
            positive_count: Number of positive testimonials
            negative_count: Number of negative testimonials

        Returns:
            List of actionable suggestions
        """
        suggestions = []
        total = len(testimonials_df)

        # Get positive and negative testimonials
        positive_testimonials = testimonials_df[testimonials_df['sentiment_label'] == 'POSITIVE']
        negative_testimonials = testimonials_df[testimonials_df['sentiment_label'] == 'NEGATIVE']

        positive_texts = positive_testimonials['text'].tolist()
        negative_texts = negative_testimonials['text'].tolist()

        # Analyze what to improve
        if negative_count > 0:
            # Extract issues from negative testimonials
            negative_keywords = self.extract_keywords(negative_texts, top_n=5)
            topics = self.identify_topics(negative_texts)

            # Priority issues
            for word, count in negative_keywords[:3]:
                if word in ['slow', 'unresponsive', 'performance']:
                    suggestions.append(f"⚡ Optimize app performance - '{word}' mentioned {count} times")
                elif word in ['confusing', 'complicated', 'difficult']:
                    suggestions.append(f"🎯 Simplify user interface - usability issues mentioned {count} times")
                elif word in ['bugs', 'crashes', 'errors', 'issues']:
                    suggestions.append(f"🐛 Fix technical issues - stability problems mentioned {count} times")
                elif word in ['support', 'service', 'help']:
                    suggestions.append(f"📞 Improve customer support - service issues mentioned {count} times")

            # Topic-based suggestions
            sorted_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)
            for topic, count in sorted_topics[:2]:
                if count > 0:
                    if topic == 'ease_of_use':
                        suggestions.append(f"💡 Add onboarding tutorial - usability mentioned {count} times")
                    elif topic == 'performance':
                        suggestions.append(f"🚀 Optimize loading times - performance mentioned {count} times")
                    elif topic == 'customer_service':
                        suggestions.append(f"🤝 Expand support channels - mentioned {count} times")

        # Leverage strengths from positive testimonials
        if positive_count > 0:
            positive_keywords = self.extract_keywords(positive_texts, top_n=5)

            for word, count in positive_keywords[:2]:
                if count >= 3:
                    if word in ['love', 'amazing', 'excellent', 'fantastic', 'best']:
                        suggestions.append(f"⭐ Feature customer testimonials on homepage - '{word}' mentioned {count} times")
                    elif word in ['easy', 'simple', 'intuitive', 'efficient']:
                        suggestions.append(f"✨ Highlight ease of use in marketing - {count} positive mentions")

        # Rating-based suggestions
        if avg_rating < 3.0:
            suggestions.append("🚨 CRITICAL: Major product improvements needed - low satisfaction")
        elif avg_rating < 4.0:
            suggestions.append("📊 Focus on addressing top customer complaints")
        elif avg_rating >= 4.5:
            suggestions.append("🎉 Leverage high satisfaction in marketing campaigns")

        # Conversion suggestions
        positive_pct = (positive_count / total * 100) if total > 0 else 0
        if positive_pct >= 85:
            suggestions.append("📣 Create case studies from happy customers")
        elif positive_pct < 60:
            suggestions.append("🔄 Implement feedback loop to address concerns quickly")

        # Engagement suggestions
        if total < 20:
            suggestions.append("📨 Increase testimonial collection through email campaigns")
        else:
            suggestions.append("💬 Continue gathering feedback to maintain customer insights")

        # Ensure we have at least some suggestions
        if not suggestions:
            suggestions.append("✅ Maintain current service quality standards")
            suggestions.append("📣 Encourage more customer testimonials")

        return suggestions[:6]  # Return top 6 actionable suggestions

    def _identify_satisfaction_drivers(
        self,
        texts: list[str],
        topics: dict[str, int]
    ) -> list[str]:
        """Identify what drives customer satisfaction."""
        drivers = []

        # Map topics to satisfaction drivers
        driver_templates = {
            'quality': 'Product quality exceeds expectations',
            'customer_service': 'Responsive and helpful customer support',
            'shipping': 'Fast and reliable delivery experience',
            'price': 'Good value for money',
            'ease_of_use': 'User-friendly products and services',
            'design': 'Attractive and thoughtful design',
            'performance': 'Products perform as advertised',
            'durability': 'Long-lasting, reliable products',
        }

        # Sort topics by frequency
        sorted_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)

        for topic, count in sorted_topics:
            if count > 0 and topic in driver_templates:
                drivers.append(f"{driver_templates[topic]} ({count} mentions)")

        if not drivers:
            drivers = ["Overall positive shopping experience"]

        return drivers

    def _generate_key_themes(
        self,
        topics: dict[str, int],
        common_phrases: list[tuple[str, int]]
    ) -> list[str]:
        """Generate key themes from analysis."""
        themes = []

        # Add topic-based themes
        for topic, count in sorted(topics.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                theme_names = {
                    'quality': 'Product Quality',
                    'price': 'Value & Pricing',
                    'shipping': 'Delivery Experience',
                    'customer_service': 'Customer Support',
                    'ease_of_use': 'User Experience',
                    'design': 'Design & Aesthetics',
                    'performance': 'Product Performance',
                    'durability': 'Durability & Reliability',
                }
                if topic in theme_names:
                    themes.append(f"{theme_names[topic]} ({count} mentions)")

        return themes

    def _identify_improvement_priorities(
        self,
        texts: list[str],
        topics: dict[str, int]
    ) -> list[str]:
        """Identify areas that need improvement."""
        priorities = []

        # Look for negative indicators in texts
        negative_patterns = [
            (r'\b(slow|delayed)\b', 'Speed up shipping and delivery times'),
            (r'\b(expensive|costly|overpriced)\b', 'Review pricing strategy'),
            (r'\b(confusing|complicated)\b', 'Simplify user experience'),
            (r'\b(poor|bad)\s+quality\b', 'Improve product quality control'),
            (r'\b(unhelpful|rude)\b', 'Enhance customer service training'),
            (r'\b(broke|broken|defective)\b', 'Address durability issues'),
        ]

        combined_text = ' '.join(texts).lower()

        for pattern, priority in negative_patterns:
            matches = re.findall(pattern, combined_text)
            if matches:
                count = len(matches)
                priorities.append(f"{priority} ({count} mentions)")

        if not priorities:
            priorities = ["No major issues identified - maintain current standards"]

        return priorities


def get_product_insights(
    product_name: str,
    reviews_df: pd.DataFrame
) -> dict[str, Any]:
    """
    Convenience function to get AI insights for a product.

    Args:
        product_name: Name of the product
        reviews_df: DataFrame with analyzed reviews

    Returns:
        Dictionary with insights
    """
    analyzer = AIInsightsAnalyzer()
    return analyzer.analyze_product_reviews(product_name, reviews_df)


def get_testimonial_insights(testimonials_df: pd.DataFrame) -> dict[str, Any]:
    """
    Convenience function to get AI insights for testimonials.

    Args:
        testimonials_df: DataFrame with testimonials

    Returns:
        Dictionary with insights
    """
    analyzer = AIInsightsAnalyzer()
    return analyzer.analyze_testimonials(testimonials_df)
