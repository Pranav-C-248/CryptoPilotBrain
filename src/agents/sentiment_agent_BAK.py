import feedparser
from dotenv import dotenv_values
from google import genai
from google.genai import types
import json
import re

class SentimentAgent:
    def __init__(self):
        config = dotenv_values(".env")
        self.client = genai.Client(api_key=config.get("gemini_key", ""))
        self.model_name = "gemini-2.5-flash"

    def fetch_and_analyze_news(self, limit=5):
        url = "https://www.coindesk.com/arc/outboundfeeds/rss/"
        feed = feedparser.parse(url)
        
        articles = []
        for entry in feed.entries[:limit]:
            prompt = f"""
            Analyze the sentiment of the following crypto news headline and output a sentiment score between 0.0 and 1.0.
            0.0 = Extreme Fear / Highly Bearish
            0.5 = Neutral
            1.0 = Extreme Greed / Highly Bullish
            
            Headline: {entry.title}
            
            Output ONLY a valid JSON object in this format:
            {{"sentiment_score": 0.xx, "reasoning": "brief explanation"}}
            """
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0,
                        response_mime_type="application/json"
                    )
                )
                parsed = json.loads(self._clean_json_response(response.text))
                score = float(parsed.get("sentiment_score", 0.5))
                reasoning = parsed.get("reasoning", "No reasoning provided.")
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

    def _clean_json_response(self, content: str) -> str:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
        if match:
            return match.group(1)
        return content.strip()
