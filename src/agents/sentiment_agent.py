import feedparser
from transformers import pipeline

class SentimentAgent:
    _pipeline = None

    def __init__(self):
        if SentimentAgent._pipeline is None:
            SentimentAgent._pipeline = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                top_k=None
            )
        self.model = SentimentAgent._pipeline

    def fetch_and_analyze_news(self, limit=5):
        url = "https://www.coindesk.com/arc/outboundfeeds/rss/"
        feed = feedparser.parse(url)

        articles = []
        for entry in feed.entries[:limit]:
            try:
                results = self.model(entry.title[:512])[0]
                scores = {r["label"]: r["score"] for r in results}

                # Map FinBERT's 3-class output to a 0.0–1.0 scale
                # positive=1.0, neutral=0.5, negative=0.0
                score = (
                    scores.get("positive", 0.0) * 1.0 +
                    scores.get("neutral",  0.0) * 0.5 +
                    scores.get("negative", 0.0) * 0.0
                )

                dominant = max(scores, key=scores.get)
                confidence = scores[dominant]
                reasoning = f"FinBERT: {dominant} ({confidence:.0%} confidence)"

            except Exception as e:
                print(f"Sentiment analysis failed for '{entry.title}': {e}")
                score = 0.5
                reasoning = "Failed to parse sentiment."

            articles.append({
                "title": entry.title,
                "link": entry.link,
                "published": entry.published,
                "sentiment_score": score,
                "reasoning": reasoning
            })

        return articles

    def get_aggregate_sentiment(self, limit=5) -> float:
        articles = self.fetch_and_analyze_news(limit)
        if not articles:
            return 0.5
        avg_score = sum(a["sentiment_score"] for a in articles) / len(articles)
        return float(avg_score)