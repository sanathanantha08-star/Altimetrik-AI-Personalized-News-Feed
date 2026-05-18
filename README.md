Prompt used to build the backend:

You are a senior backend engineer, generate a complete production grade fastAPI backedn for an AI news feed personalizer. Follow every rule below exactly.
Architecture and code quality: -Folllow the clean code architecture which use routers, services, repositories,models.

* routers call service only
* repositories contain raw db queries only
* never hardcode sensitive values
-have a core folder which will include the config,security,logging,excwpotions and retries.

* have a db folder for the get_db dependency , and wrap it in a main file . use python and fast api for this.
* Store all sesnitive info in the .env file and call them from the config settings file, examples of env variabkles are newsapi key, news url endpoint, mongodburi,mongodbname,ratelimitdefault, cors origins,logl_level. User types: There will be only one user and do not create any routes or apis for user registration and login, we dont ened auth in this app, just need to store the user porefernces which i will copme later on to.
Mongodb collections and schemas: User Prefernce db: interests: List[str]





# AI News Feed Personalizer — Backend

A production-grade FastAPI backend that delivers a personalised news feed using:

- **Hybrid Retrieval** — BM25 (lexical) + Semantic search (Cohere embeddings), fused via Reciprocal Rank Fusion
- **Query Expansion** — Cohere LLM expands user interests into richer search terms
- **Cross-Encoder Reranking** — FlashRank (ms-marco-MiniLM-L-12-v2, runs locally)
- **LLM Personalisation Reasoning** — Cohere command-r-plus explains why each article matches
- **APScheduler** — recurring news ingest from NewsAPI every N minutes
- **MongoDB** — stores raw articles + dense embeddings + BM25 tokens

---

## Project Structure

```
ai-news-feed/
├── core/
│   ├── config.py                   # Pydantic Settings (reads .env)
│   ├── exceptions.py               # Domain exception hierarchy
│   ├── logging.py                  # structlog + RequestTracingMiddleware
│   ├── retries.py                  # Tenacity retry decorators
│   ├── security.py                 # Input sanitisation
│   └── __init__.py
│
├── db/
│   ├── session.py                  # Motor client, connect/close, get_db
│   └── __init__.py
│
├── app/
│   ├── models/
│   │   ├── user_preference.py      # Singleton user prefs document
│   │   └── news_article.py         # Vector store document
│   ├── schemas/
│   │   └── user_preference.py      # HTTP request/response DTOs
│   ├── repositories/
│   │   ├── user_preference_repository.py
│   │   └── news_article_repository.py
│   ├── services/
│   │   ├── user_preference_service.py
│   │   ├── news_ingest_service.py   # Fetch → embed → store
│   │   ├── embedding_service.py    # Cohere embed + cosine similarity
│   │   ├── bm25_service.py         # BM25 lexical retrieval
│   │   ├── reranker_service.py     # FlashRank cross-encoder
│   │   ├── query_expansion_service.py
│   │   ├── cohere_llm_service.py   # Personalisation reasoning
│   │   └── feed_service.py         # Full pipeline orchestration
│   ├── routers/
│   │   ├── preferences.py          # GET/POST /user-preference
│   │   ├── scheduler.py            # POST /scheduler/run
│   │   └── feed.py                 # GET /feed
│   └── app.py                      # FastAPI factory
│
├── tests/
│   ├── test_feed_service.py
│   └── test_user_preference_service.py
│
├── main.py                         # Entry point + lifespan + APScheduler
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── .env.example
└── .gitignore
```

---

## Setup

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your real values

# 4. Run
python main.py
# → http://localhost:8000/docs
```

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `NEWS_API_KEY` | NewsAPI key from newsapi.org | required |
| `NEWS_API_TOP_HEADLINES_ENDPOINT` | NewsAPI endpoint | https://newsapi.org/v2/top-headlines |
| `NEWS_FETCH_COUNTRY` | Country code for headlines | `us` |
| `MONGODB_URI` | MongoDB connection string | required |
| `MONGODB_NAME` | Database name | required |
| `COHERE_API_KEY` | Cohere API key | required |
| `COHERE_MODEL` | Cohere chat model | `command-r-plus-08-2024` |
| `COHERE_EMBED_MODEL` | Cohere embed model | `embed-english-v3.0` |
| `SIMILARITY_THRESHOLD` | Min cosine score to include article | `0.30` |
| `TOP_K_SEMANTIC` | Max semantic candidates | `20` |
| `TOP_K_BM25` | Max BM25 candidates | `20` |
| `TOP_K_RERANK` | Max articles after reranking | `10` |
| `SCHEDULER_INTERVAL_MINUTES` | Ingest interval | `30` |
| `RATE_LIMIT_DEFAULT` | slowapi rate limit | `60/minute` |
| `CORS_ORIGINS` | JSON array of allowed origins | `["http://localhost:3000"]` |
| `LOG_LEVEL` | Log verbosity | `INFO` |
| `APP_ENV` | `development` or `production` | `development` |

---

## API Reference

### User Preferences

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/user-preference` | Get saved interests |
| `POST` | `/api/v1/user-preference` | Save / replace interests |

**POST body:**
```json
{ "interests": ["AI", "cloud", "startups"] }
```

### Scheduler

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/scheduler/run` | Manually trigger news ingest |

### Feed

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/feed` | Personalized feed |

**Sample response item:**
```json
{
  "article": {
    "id": "https://...",
    "title": "New AI startup raises funding",
    "description": "...",
    "url": "https://...",
    "source_name": "TechCrunch",
    "published_at": "2026-05-18T10:00:00Z"
  },
  "score": 0.8921,
  "semantic_score": 0.7843,
  "bm25_score": 0.9200,
  "rerank_score": 0.9450,
  "rank": 1,
  "reason": "Matches AI and startups interests.",
  "matched_interests": ["AI", "startups"]
}
```

### Health

| Method | Path |
|---|---|
| `GET` | `/health` |

---

## Feed Pipeline

```
User interests
     │
     ▼
Query Expansion (Cohere LLM)
     │
     ├──────────────────────────────────┐
     ▼                                  ▼
Semantic Search                     BM25 Search
(Cohere embeddings + cosine sim)   (rank-bm25, in-memory)
     │                                  │
     └──────────┐  ┌─────────────────────┘
                ▼  ▼
         Reciprocal Rank Fusion (RRF)
                │
                ▼
         Filter by threshold
                │
                ▼
       FlashRank Cross-encoder Reranking
                │
                ▼
       Cohere LLM Personalisation Reasoning
                │
                ▼
    Final sort: rerank_score → llm_score (tie-break)
                │
                ▼
         FeedItemResponse[]
```

---

## Architecture Rules

- **Routers** → call services only
- **Services** → business logic, call repositories
- **Repositories** → raw MongoDB queries only
- **Models** → MongoDB document shapes
- **Schemas** → HTTP DTOs
- **Core** → zero app imports; only stdlib + third-party

---

## Running Tests

```bash
pip install -r requirements-dev.txt
pytest -v
```
