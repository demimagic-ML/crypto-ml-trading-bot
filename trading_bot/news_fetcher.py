"""
News Fetcher with API + Qwen Vision fallback.
Fetches crypto news from multiple sources and analyzes sentiment.
"""
import os
import sys
import json
import time
import base64
import requests
from datetime import datetime, timedelta
from io import BytesIO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

class NewsFetcher:
    def __init__(self):
        self.dashscope_key = os.getenv('DASHSCOPE_API_KEY', '')
        self.news_cache = []
        self.last_fetch = None
        self.cache_duration = 300
        
        self.news_dir = os.path.join(os.path.dirname(__file__), 'data', 'news')
        os.makedirs(self.news_dir, exist_ok=True)
        
        self.screenshots_dir = os.path.join(os.path.dirname(__file__), 'data', 'screenshots')
        os.makedirs(self.screenshots_dir, exist_ok=True)
        
        self.screenshot_cache = {}
        self.screenshot_cache_duration = 600
        
        self.news_sources = {
            'cryptopanic': 'https://cryptopanic.com/api/v1/posts/',
            'newsapi': 'https://newsapi.org/v2/everything',
        }
        
        self.vision_sites = [
            {'url': 'https://www.coindesk.com/', 'type': 'news', 'name': 'coindesk'},
            {'url': 'https://cointelegraph.com/', 'type': 'news', 'name': 'cointelegraph'},
            {'url': 'https://decrypt.co/', 'type': 'news', 'name': 'decrypt'},
        ]
        
        self.macro_rss_feeds = [
            {'url': 'https://www.federalreserve.gov/feeds/press_all.xml', 'name': 'fed_press', 'type': 'fed'},
            {'url': 'https://feeds.reuters.com/reuters/businessNews', 'name': 'reuters_business', 'type': 'macro'},
            {'url': 'https://feeds.reuters.com/reuters/topNews', 'name': 'reuters_top', 'type': 'macro'},
            {'url': 'https://www.cnbc.com/id/10001147/device/rss/rss.html', 'name': 'cnbc_markets', 'type': 'macro'},
            {'url': 'https://www.cnbc.com/id/20910258/device/rss/rss.html', 'name': 'cnbc_economy', 'type': 'macro'},
            {'url': 'https://feeds.marketwatch.com/marketwatch/topstories/', 'name': 'marketwatch', 'type': 'macro'},
            {'url': 'https://finance.yahoo.com/news/rssindex', 'name': 'yahoo_finance', 'type': 'macro'},
            {'url': 'https://feeds.bloomberg.com/markets/news.rss', 'name': 'bloomberg_markets', 'type': 'macro'},
        ]
        
        self.macro_keywords = [
            'fed', 'federal reserve', 'interest rate', 'inflation', 'cpi', 'ppi',
            'fomc', 'powell', 'treasury', 'bond', 'yield', 'dollar', 'dxy',
            'recession', 'gdp', 'unemployment', 'jobs', 'nonfarm', 'payroll',
            'bitcoin', 'crypto', 'digital asset', 'etf', 'sec', 'regulation',
            'risk', 'liquidity', 'qe', 'qt', 'taper', 'rate cut', 'rate hike',
            'bank', 'financial', 'stock', 'market', 'rally', 'crash', 'volatility'
        ]
    
    def _get_date_folder(self):
        """Get/create folder for today's date."""
        today = datetime.now().strftime('%Y-%m-%d')
        date_folder = os.path.join(self.news_dir, today)
        os.makedirs(date_folder, exist_ok=True)
        return date_folder
    
    def _save_articles(self, articles: list, source_type: str):
        """Save articles to date folder, organized by source type."""
        if not articles:
            return
        
        date_folder = self._get_date_folder()
        type_folder = os.path.join(date_folder, source_type)
        os.makedirs(type_folder, exist_ok=True)
        
        timestamp = datetime.now().strftime('%H%M%S')
        filename = f"{timestamp}_articles.json"
        filepath = os.path.join(type_folder, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'fetch_time': datetime.now().isoformat(),
                'source_type': source_type,
                'count': len(articles),
                'articles': articles
            }, f, indent=2, ensure_ascii=False)
        
        print(f"  [Save] Saved {len(articles)} articles to {source_type}/{filename}")
    
    def _save_analysis(self, analysis: dict):
        """Save sentiment analysis result."""
        date_folder = self._get_date_folder()
        analysis_folder = os.path.join(date_folder, 'analysis')
        os.makedirs(analysis_folder, exist_ok=True)
        
        timestamp = datetime.now().strftime('%H%M%S')
        filename = f"{timestamp}_sentiment.json"
        filepath = os.path.join(analysis_folder, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        print(f"  [Save] Saved analysis to analysis/{filename}")
    
    def fetch_all_news(self) -> dict:
        """Fetch news from all sources and return aggregated sentiment."""
        if self.last_fetch and (datetime.now() - self.last_fetch).seconds < self.cache_duration:
            return self._analyze_cached_news()
        
        all_news = []
        
        cryptopanic_news = self._fetch_cryptopanic()
        if cryptopanic_news:
            self._save_articles(cryptopanic_news, 'cryptopanic')
            all_news.extend(cryptopanic_news)
        
        rss_news = self._fetch_rss_feeds()
        if rss_news:
            self._save_articles(rss_news, 'rss_feeds')
            all_news.extend(rss_news)
        
        macro_news = self._fetch_macro_news()
        if macro_news:
            self._save_articles(macro_news, 'macro_news')
            all_news.extend(macro_news)
        
        if SELENIUM_AVAILABLE:
            vision_news = self._fetch_with_vision()
            if vision_news:
                self._save_articles(vision_news, 'vision')
                all_news.extend(vision_news)
        
        self.news_cache = all_news
        self.last_fetch = datetime.now()
        
        analysis = self._analyze_news(all_news)
        self._save_analysis(analysis)
        
        return analysis
    
    def _fetch_cryptopanic(self) -> list:
        """Fetch from CryptoPanic API (free tier)."""
        try:
            url = "https://cryptopanic.com/api/v1/posts/?auth_token=free&currencies=BTC&filter=hot"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                news = []
                for item in data.get('results', [])[:10]:
                    news.append({
                        'source': 'cryptopanic',
                        'title': item.get('title', ''),
                        'published': item.get('published_at', ''),
                        'url': item.get('url', ''),
                        'votes': item.get('votes', {}),
                    })
                print(f"  [CryptoPanic] Fetched {len(news)} articles")
                return news
        except Exception as e:
            print(f"  [CryptoPanic] Error: {e}")
        return []
    
    def _fetch_rss_feeds(self) -> list:
        """Fetch from crypto RSS feeds."""
        try:
            import feedparser
        except ImportError:
            return []
        
        feeds = [
            'https://cointelegraph.com/rss',
            'https://www.coindesk.com/arc/outboundfeeds/rss/',
        ]
        
        news = []
        for feed_url in feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:5]:
                    title = entry.get('title', '').lower()
                    if 'bitcoin' in title or 'btc' in title or 'crypto' in title:
                        news.append({
                            'source': feed_url.split('/')[2],
                            'title': entry.get('title', ''),
                            'published': entry.get('published', ''),
                            'summary': entry.get('summary', '')[:200],
                            'type': 'crypto'
                        })
                print(f"  [RSS] Fetched {len(news)} articles")
            except Exception as e:
                print(f"  [RSS] Error with {feed_url}: {e}")
        
        return news
    
    def _fetch_macro_news(self) -> list:
        """Fetch macro/financial news from Fed, Reuters, CNBC, etc."""
        try:
            import feedparser
        except ImportError:
            print("  [Macro] feedparser not available")
            return []
        
        news = []
        fed_count = 0
        macro_count = 0
        
        for feed_info in self.macro_rss_feeds:
            try:
                feed = feedparser.parse(feed_info['url'])
                for entry in feed.entries[:10]:
                    title = entry.get('title', '').lower()
                    summary = entry.get('summary', '').lower()
                    content = title + ' ' + summary
                    
                    is_relevant = any(kw in content for kw in self.macro_keywords)
                    
                    if is_relevant:
                        news_item = {
                            'source': feed_info['name'],
                            'title': entry.get('title', ''),
                            'published': entry.get('published', ''),
                            'summary': entry.get('summary', '')[:300],
                            'type': feed_info['type'],
                            'is_fed': feed_info['type'] == 'fed'
                        }
                        news.append(news_item)
                        
                        if feed_info['type'] == 'fed':
                            fed_count += 1
                        else:
                            macro_count += 1
                            
            except Exception as e:
                pass
        
        if fed_count > 0 or macro_count > 0:
            print(f"  [Macro] Fed: {fed_count} | Financial: {macro_count} articles")
        
        return news[:15]
    
    def _save_screenshot(self, screenshot_bytes: bytes, site_name: str, site_type: str) -> str:
        """Save screenshot to folder and return path."""
        today = datetime.now().strftime('%Y-%m-%d')
        timestamp = datetime.now().strftime('%H%M%S')
        
        type_folder = os.path.join(self.screenshots_dir, today, site_type)
        os.makedirs(type_folder, exist_ok=True)
        
        filename = f"{timestamp}_{site_name}.png"
        filepath = os.path.join(type_folder, filename)
        
        with open(filepath, 'wb') as f:
            f.write(screenshot_bytes)
        
        print(f"  [Screenshot] Saved to {site_type}/{filename}")
        return filepath
    
    def _fetch_with_vision(self) -> list:
        """Screenshot news sites and Reddit forums, use Qwen vision to extract sentiment."""
        if not SELENIUM_AVAILABLE:
            print("  [Vision] Selenium not available")
            return []
        
        news = []
        
        try:
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--window-size=1920,1200')
            options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
            
            driver = webdriver.Chrome(options=options)
            
            sites_to_check = self.vision_sites[:3]
            
            for site_info in sites_to_check:
                try:
                    url = site_info['url']
                    site_type = site_info['type']
                    site_name = site_info['name']
                    
                    if site_name in self.screenshot_cache:
                        cached = self.screenshot_cache[site_name]
                        age = (datetime.now() - cached['time']).seconds
                        if age < self.screenshot_cache_duration:
                            print(f"  [Vision] Using cached {site_name} ({age}s old)")
                            if cached.get('analysis'):
                                news.append(cached['analysis'])
                            continue
                    
                    print(f"  [Vision] Screenshotting {site_name} ({site_type})...")
                    driver.get(url)
                    time.sleep(4)
                    
                    driver.execute_script("window.scrollTo(0, 500);")
                    time.sleep(1)
                    
                    screenshot = driver.get_screenshot_as_png()
                    
                    self._save_screenshot(screenshot, site_name, site_type)
                    
                    analysis = self._analyze_screenshot(screenshot, site_name, site_type)
                    
                    self.screenshot_cache[site_name] = {
                        'time': datetime.now(),
                        'analysis': analysis
                    }
                    
                    if analysis:
                        news.append(analysis)
                        
                except Exception as e:
                    print(f"  [Vision] Error with {site_name}: {e}")
            
            driver.quit()
            
        except Exception as e:
            print(f"  [Vision] Driver error: {e}")
        
        return news
    
    def _analyze_screenshot(self, screenshot_bytes: bytes, site_name: str, site_type: str) -> dict:
        """Use Qwen vision to analyze a screenshot based on site type."""
        if not self.dashscope_key:
            return None
        
        try:
            img_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
            
            url = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
            
            headers = {
                "Authorization": f"Bearer {self.dashscope_key}",
                "Content-Type": "application/json"
            }
            
            if site_type == 'reddit':
                prompt_text = f"""Analyze this Reddit cryptocurrency forum screenshot ({site_name}).

You are analyzing community sentiment from Reddit posts and comments.

EXTRACT:
1. Top 3-5 post titles or discussion topics visible
2. Overall community sentiment: BULLISH, BEARISH, or NEUTRAL
3. Key themes being discussed (price predictions, news reactions, FUD, FOMO)
4. Emotional tone: excited, fearful, confident, uncertain

TRADING INSIGHT:
- Reddit sentiment often leads price moves by 1-4 hours
- High excitement = potential local top
- Extreme fear = potential buying opportunity
- Look for unusual activity or breaking news discussions

RESPOND IN JSON:
{{
    "source": "{site_name}",
    "source_type": "reddit",
    "posts": ["post1", "post2", "post3"],
    "sentiment": "BULLISH/BEARISH/NEUTRAL",
    "sentiment_score": 0.0,
    "confidence": 0.8,
    "community_mood": "excited/fearful/confident/uncertain/mixed",
    "key_themes": ["theme1", "theme2"],
    "trading_suggestion": "STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL",
    "urgency": "HIGH/MEDIUM/LOW",
    "summary": "brief summary of community sentiment"
}}"""
            else:
                prompt_text = f"""Analyze this crypto news website screenshot ({site_name}).

You are a trading analyst extracting actionable intelligence from news.

EXTRACT:
1. Top 3 headlines about Bitcoin/crypto
2. Overall sentiment: BULLISH, BEARISH, or NEUTRAL  
3. Any major events (ETF, regulation, hack, adoption, institutional moves)
4. Breaking vs old news

TRADING RULES:
- Breaking news > recycled news
- "Buy the rumor, sell the news"
- Institutional moves are high impact
- Regulatory news can cause 5%+ moves

RESPOND IN JSON:
{{
    "source": "{site_name}",
    "source_type": "news",
    "headlines": ["headline1", "headline2", "headline3"],
    "sentiment": "BULLISH/BEARISH/NEUTRAL",
    "sentiment_score": 0.0,
    "confidence": 0.8,
    "events": ["event1", "event2"],
    "is_breaking_news": true/false,
    "market_impact": "SIGNIFICANT/MODERATE/MINIMAL",
    "trading_suggestion": "STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL",
    "urgency": "HIGH/MEDIUM/LOW",
    "summary": "brief summary"
}}"""
            
            payload = {
                "model": "qwen-vl-max",
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"image": f"data:image/png;base64,{img_base64}"},
                                {"text": prompt_text}
                            ]
                        }
                    ]
                }
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                output = result.get('output', {})
                
                if 'text' in output:
                    content = output['text']
                elif 'choices' in output:
                    content = output['choices'][0].get('message', {}).get('content', '')
                else:
                    content = str(output)
                
                if isinstance(content, list):
                    content = ' '.join([str(c.get('text', c)) if isinstance(c, dict) else str(c) for c in content])
                
                try:
                    start = content.find('{')
                    end = content.rfind('}') + 1
                    if start >= 0 and end > start:
                        analysis = json.loads(content[start:end])
                        analysis['source'] = site_name
                        print(f"  [Vision] Analyzed {site_name}: {analysis.get('sentiment', 'N/A')}")
                        return analysis
                except json.JSONDecodeError:
                    print(f"  [Vision] Could not parse JSON from response")
            else:
                print(f"  [Vision] API error: {response.status_code}")
                
        except Exception as e:
            print(f"  [Vision] Analysis error: {e}")
        
        return None
    
    def _analyze_news(self, news_list: list) -> dict:
        """Analyze collected news using Qwen for sentiment."""
        if not news_list:
            return {
                'sentiment': 'NEUTRAL',
                'sentiment_score': 0.0,
                'confidence': 0,
                'news_count': 0,
                'headlines': [],
                'events': [],
                'timestamp': datetime.now().isoformat()
            }
        
        headlines = [n.get('title', '') or n.get('headlines', [''])[0] for n in news_list[:10]]
        headlines = [h for h in headlines if h]
        
        vision_news = [n for n in news_list if 'sentiment_score' in n]
        if vision_news:
            avg_score = sum(n['sentiment_score'] for n in vision_news) / len(vision_news)
            events = []
            for n in vision_news:
                events.extend(n.get('events', []))
            
            if avg_score > 0.5:
                suggestion = 'STRONG_BUY'
            elif avg_score > 0.2:
                suggestion = 'BUY'
            elif avg_score < -0.5:
                suggestion = 'STRONG_SELL'
            elif avg_score < -0.2:
                suggestion = 'SELL'
            else:
                suggestion = 'HOLD'
            
            return {
                'sentiment': 'BULLISH' if avg_score > 0.2 else ('BEARISH' if avg_score < -0.2 else 'NEUTRAL'),
                'sentiment_score': avg_score,
                'confidence': 0.6,
                'trading_suggestion': suggestion,
                'news_count': len(news_list),
                'headlines': headlines[:5],
                'events': list(set(events)),
                'timestamp': datetime.now().isoformat()
            }
        
        if self.dashscope_key and headlines:
            return self._analyze_headlines_with_qwen(headlines)
        
        return self._simple_sentiment_analysis(headlines)
    
    def _analyze_headlines_with_qwen(self, headlines: list) -> dict:
        """Use Qwen VL model to analyze headlines."""
        try:
            url = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
            
            headers = {
                "Authorization": f"Bearer {self.dashscope_key}",
                "Content-Type": "application/json"
            }
            
            headlines_text = "\n".join([f"- {h}" for h in headlines])
            
            prompt = f"""You are an expert Bitcoin trading analyst. Analyze these news headlines for Bitcoin impact:

{headlines_text}

MACRO/FED NEWS IMPACT ON BITCOIN:
- Rate HIKE / Hawkish Fed = BEARISH for Bitcoin (risk-off, strong dollar)
- Rate CUT / Dovish Fed = BULLISH for Bitcoin (risk-on, weak dollar)
- High Inflation CPI = Mixed (short-term bearish, long-term bullish as hedge)
- Recession fears = BEARISH short-term (risk-off)
- Bank failures / Financial stress = BULLISH (Bitcoin as safe haven)
- Strong jobs/economy = BEARISH (Fed stays hawkish)
- Weak jobs/economy = BULLISH (Fed may cut rates)
- Dollar strength (DXY up) = BEARISH for Bitcoin
- Dollar weakness = BULLISH for Bitcoin

CRYPTO-SPECIFIC:
- ETF approval/inflows = BULLISH
- Regulatory crackdown = BEARISH
- Exchange hacks/failures = BEARISH
- Institutional adoption = BULLISH

RESPOND IN THIS EXACT JSON FORMAT:
{{"sentiment": "BULLISH" or "BEARISH" or "NEUTRAL", "sentiment_score": -1.0 to 1.0, "confidence": 0.0 to 1.0, "key_events": ["event1"], "trading_suggestion": "STRONG_BUY" or "BUY" or "HOLD" or "SELL" or "STRONG_SELL"}}"""
            
            payload = {
                "model": "qwen-vl-max",
                "input": {
                    "messages": [
                        {"role": "user", "content": [{"text": prompt}]}
                    ]
                }
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                content = ""
                if 'output' in result:
                    output = result['output']
                    choices = output.get('choices', [])
                    if choices:
                        msg_content = choices[0].get('message', {}).get('content', [])
                        if isinstance(msg_content, list):
                            for item in msg_content:
                                if isinstance(item, dict) and 'text' in item:
                                    content = item['text']
                                    break
                        elif isinstance(msg_content, str):
                            content = msg_content
                    if not content:
                        content = output.get('text', '')
                
                start = content.find('{')
                end = content.rfind('}') + 1
                if start >= 0 and end > start:
                    analysis = json.loads(content[start:end])
                    analysis['headlines'] = headlines[:5]
                    analysis['news_count'] = len(headlines)
                    analysis['timestamp'] = datetime.now().isoformat()
                    print(f"  [Qwen] Sentiment: {analysis.get('sentiment')} ({analysis.get('sentiment_score')})")
                    return analysis
                    
        except Exception as e:
            print(f"  [Qwen] Analysis error: {e}")
        
        return self._simple_sentiment_analysis(headlines)
    
    def _simple_sentiment_analysis(self, headlines: list) -> dict:
        """Fallback: simple keyword-based sentiment."""
        bullish_words = ['surge', 'rally', 'bullish', 'gains', 'soars', 'breakout', 'etf approved', 'adoption', 'buy']
        bearish_words = ['crash', 'drop', 'bearish', 'plunge', 'sell-off', 'hack', 'ban', 'regulation', 'fear']
        
        score = 0
        for headline in headlines:
            h_lower = headline.lower()
            for word in bullish_words:
                if word in h_lower:
                    score += 1
            for word in bearish_words:
                if word in h_lower:
                    score -= 1
        
        if headlines:
            score = max(-1, min(1, score / len(headlines)))
        
        return {
            'sentiment': 'BULLISH' if score > 0.2 else ('BEARISH' if score < -0.2 else 'NEUTRAL'),
            'sentiment_score': score,
            'confidence': 0.5 if headlines else 0,
            'news_count': len(headlines),
            'headlines': headlines[:5],
            'events': [],
            'timestamp': datetime.now().isoformat()
        }
    
    def _analyze_cached_news(self) -> dict:
        """Return analysis of cached news."""
        return self._analyze_news(self.news_cache)

def test_news_fetcher():
    """Test the news fetcher."""
    print("="*50)
    print("NEWS FETCHER TEST")
    print("="*50)
    
    fetcher = NewsFetcher()
    result = fetcher.fetch_all_news()
    
    print("\n" + "="*50)
    print("RESULTS:")
    print("="*50)
    print(f"Sentiment: {result.get('sentiment')}")
    print(f"Score: {result.get('sentiment_score')}")
    print(f"News Count: {result.get('news_count')}")
    print(f"Headlines:")
    for h in result.get('headlines', []):
        print(f"  - {h[:80]}...")
    print(f"Events: {result.get('events', [])}")
    
    return result

if __name__ == '__main__':
    test_news_fetcher()
