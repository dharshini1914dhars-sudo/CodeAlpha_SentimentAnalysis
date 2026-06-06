# ============================================================
# CodeAlpha Internship — Task 4: Sentiment Analysis
# Dataset: Amazon Product Reviews
# Author: [Your Name]
# ============================================================

# Step 1: Install required libraries
# pip install pandas numpy matplotlib seaborn textblob wordcloud

# Step 2: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="darkgrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ============================================================
# Step 3: Create Sample Amazon Reviews Dataset
# ============================================================
data = {
    'review': [
        "This product is absolutely amazing! Best purchase ever!",
        "Terrible quality, broke after one day. Very disappointed.",
        "Okay product, nothing special. Does the job.",
        "Excellent! Exceeded my expectations. Highly recommend!",
        "Worst product I have ever bought. Complete waste of money.",
        "Pretty good quality for the price. Happy with purchase.",
        "Not bad, not great. Average product overall.",
        "Fantastic! Love everything about this product!",
        "Poor quality and bad customer service. Never buying again.",
        "Great value for money! Works perfectly fine.",
        "Disappointed with the quality. Expected much better.",
        "Outstanding product! Will definitely buy again!",
        "It is okay. Nothing extraordinary about it.",
        "Horrible experience. Product stopped working in a week.",
        "Very satisfied with this purchase. Great product!",
        "Decent product. Could be better but works fine.",
        "Absolutely love it! Perfect in every way!",
        "Waste of money. Do not recommend at all.",
        "Good product overall. Minor issues but satisfied.",
        "Superb quality! Exactly as described. Very happy!",
        "Mediocre at best. Not worth the price.",
        "Brilliant product! Works like a charm!",
        "Very bad quality. Broke on first use.",
        "Amazing product! Exceeded all my expectations!",
        "Not impressed. Expected more for the price.",
        "Wonderful! So happy with this purchase!",
        "Cheap quality. Falls apart easily.",
        "Perfect product! Exactly what I needed!",
        "Okay but could be improved significantly.",
        "Love this product! Highly recommend to everyone!"
    ],
    'rating': [5, 1, 3, 5, 1, 4, 3, 5, 1, 4, 2, 5, 3, 1, 5, 3, 5, 1, 4, 5, 2, 5, 1, 5, 2, 5, 1, 5, 3, 5]
}

df = pd.DataFrame(data)

print("=" * 60)
print("AMAZON REVIEWS - SENTIMENT ANALYSIS")
print("=" * 60)
print(f"\n📌 Total Reviews: {len(df)}")
print("\n📌 Sample Reviews:")
print(df.head())

# ============================================================
# Step 4: Sentiment Analysis using TextBlob
# ============================================================
def get_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0.1:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

def get_polarity(text):
    return TextBlob(text).sentiment.polarity

def get_subjectivity(text):
    return TextBlob(text).sentiment.subjectivity

# Apply sentiment analysis
df['polarity'] = df['review'].apply(get_polarity)
df['subjectivity'] = df['review'].apply(get_subjectivity)
df['sentiment'] = df['review'].apply(get_sentiment)

print("\n📌 Sentiment Analysis Results:")
print(df[['review', 'rating', 'sentiment', 'polarity']].to_string())

# ============================================================
# Step 5: Visualizations
# ============================================================

# Chart 1: Sentiment Distribution
plt.figure(figsize=(8, 5))
colors = {'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#f39c12'}
ax = sns.countplot(x='sentiment', data=df, 
                   order=['Positive', 'Neutral', 'Negative'],
                   palette=colors)
plt.title('Sentiment Distribution of Amazon Reviews', fontsize=16, fontweight='bold')
plt.xlabel('Sentiment')
plt.ylabel('Number of Reviews')
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('sentiment_distribution.png', dpi=150)
plt.show()
print("✅ Saved: sentiment_distribution.png")

# Chart 2: Polarity Distribution
plt.figure(figsize=(10, 5))
sns.histplot(df['polarity'], bins=15, kde=True, color='steelblue')
plt.title('Polarity Score Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Polarity Score (-1=Negative, 0=Neutral, +1=Positive)')
plt.ylabel('Count')
plt.axvline(x=0, color='red', linestyle='--', label='Neutral')
plt.legend()
plt.tight_layout()
plt.savefig('polarity_distribution.png', dpi=150)
plt.show()
print("✅ Saved: polarity_distribution.png")

# Chart 3: Sentiment vs Rating
plt.figure(figsize=(10, 5))
sns.boxplot(x='sentiment', y='rating', data=df,
            order=['Positive', 'Neutral', 'Negative'],
            palette=colors)
plt.title('Sentiment vs Star Rating', fontsize=16, fontweight='bold')
plt.xlabel('Sentiment')
plt.ylabel('Star Rating')
plt.tight_layout()
plt.savefig('sentiment_vs_rating.png', dpi=150)
plt.show()
print("✅ Saved: sentiment_vs_rating.png")

# Chart 4: Polarity vs Subjectivity
plt.figure(figsize=(10, 6))
scatter_colors = df['sentiment'].map(colors)
plt.scatter(df['polarity'], df['subjectivity'], 
            c=scatter_colors, alpha=0.7, s=100, edgecolors='black')
plt.title('Polarity vs Subjectivity', fontsize=16, fontweight='bold')
plt.xlabel('Polarity (Negative → Positive)')
plt.ylabel('Subjectivity (Objective → Subjective)')
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2ecc71', label='Positive'),
                   Patch(facecolor='#f39c12', label='Neutral'),
                   Patch(facecolor='#e74c3c', label='Negative')]
plt.legend(handles=legend_elements)
plt.tight_layout()
plt.savefig('polarity_vs_subjectivity.png', dpi=150)
plt.show()
print("✅ Saved: polarity_vs_subjectivity.png")

# Chart 5: Rating Distribution
plt.figure(figsize=(8, 5))
ax = sns.countplot(x='rating', data=df, palette='viridis')
plt.title('Star Rating Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Star Rating')
plt.ylabel('Count')
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=12)
plt.tight_layout()
plt.savefig('rating_distribution.png', dpi=150)
plt.show()
print("✅ Saved: rating_distribution.png")

# Chart 6: WordCloud - Positive Reviews
positive_reviews = ' '.join(df[df['sentiment'] == 'Positive']['review'])
wordcloud_pos = WordCloud(width=800, height=400, 
                          background_color='white',
                          colormap='Greens',
                          max_words=50).generate(positive_reviews)
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud_pos, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Positive Reviews', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('wordcloud_positive.png', dpi=150)
plt.show()
print("✅ Saved: wordcloud_positive.png")

# Chart 7: WordCloud - Negative Reviews
negative_reviews = ' '.join(df[df['sentiment'] == 'Negative']['review'])
wordcloud_neg = WordCloud(width=800, height=400,
                          background_color='white',
                          colormap='Reds',
                          max_words=50).generate(negative_reviews)
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud_neg, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Negative Reviews', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('wordcloud_negative.png', dpi=150)
plt.show()
print("✅ Saved: wordcloud_negative.png")

# ============================================================
# Step 6: Key Insights
# ============================================================
print("\n" + "=" * 60)
print("📊 KEY INSIGHTS FROM SENTIMENT ANALYSIS")
print("=" * 60)

sentiment_counts = df['sentiment'].value_counts()
total = len(df)

print(f"\n1. Total Reviews Analyzed: {total}")
print(f"\n2. Sentiment Breakdown:")
for sentiment, count in sentiment_counts.items():
    print(f"   {sentiment}: {count} reviews ({count/total*100:.1f}%)")

print(f"\n3. Average Polarity Score: {df['polarity'].mean():.3f}")
print(f"   (Positive = above 0, Negative = below 0)")

print(f"\n4. Average Subjectivity Score: {df['subjectivity'].mean():.3f}")
print(f"   (0 = Objective, 1 = Subjective)")

print(f"\n5. Average Star Rating: {df['rating'].mean():.1f} / 5")

avg_polarity_by_sentiment = df.groupby('sentiment')['polarity'].mean()
print(f"\n6. Average Polarity by Sentiment:")
for sentiment, polarity in avg_polarity_by_sentiment.items():
    print(f"   {sentiment}: {polarity:.3f}")

print("\n" + "=" * 60)
print("✅ Sentiment Analysis Complete! All charts saved as PNG.")
print("=" * 60)
print("\n📌 Next Steps:")
print("  1. Upload this code to GitHub repo: CodeAlpha_SentimentAnalysis")
print("  2. Record a video explanation")
print("  3. Post on LinkedIn tagging @CodeAlpha")
print("  4. Submit via the WhatsApp group Submission Form")
