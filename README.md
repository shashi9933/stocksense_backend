# StockSense FastAPI Backend

Professional stock analysis API built with FastAPI and Python.

## 📋 Project Structure

```
backend/
├── main.py                      # Main application entry point
├── requirements.txt             # Python dependencies
└── ../utils/                    # Existing Python utilities
    ├── data_fetcher.py          # Stock data fetching
    ├── technical_indicators.py  # Indicator calculations
    ├── prediction_models.py     # ML predictions
    └── ...other utilities
```

## 🚀 Setup Instructions

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file:
```
ALPHA_VANTAGE_API_KEY=your_api_key_here
DATABASE_URL=sqlite:///./test.db
```

### 3. Run the Server
```bash
python main.py
```

Server will start at `http://localhost:8000`

### 4. View API Documentation
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 📡 API Endpoints

### Stock Data
```
GET /api/stock/{symbol}
  ?market=US&days=365
```

### Technical Indicators
```
GET /api/indicators/{symbol}
  ?market=US&days=365
```

### Price Predictions
```
GET /api/predict/{symbol}
  ?market=US&days=30
```

### Market Overview
```
GET /api/market-overview
```

### Popular Stocks
```
GET /api/popular-stocks
```

### Search
```
GET /api/search?query=apple
```

## 🔧 Environment Variables

```env
# API Keys
ALPHA_VANTAGE_API_KEY=YOUR_KEY
IEX_CLOUD_API_KEY=YOUR_KEY

# Database
DATABASE_URL=sqlite:///./stocksense.db

# Server
CORS_ORIGINS=http://localhost:5173,http://localhost:3000

# API Rate Limiting
RATE_LIMIT_CALLS=5
RATE_LIMIT_PERIOD=60
```

## 📦 Dependencies

- **FastAPI** - Web framework
- **Uvicorn** - ASGI server
- **Pandas** - Data manipulation
- **yfinance** - Stock data
- **Alpha Vantage** - Alternative data source
- **TA-Lib** - Technical analysis
- **scikit-learn** - ML models
- **TensorFlow** - Deep learning

## 🔗 CORS Configuration

Frontend can connect from:
- `http://localhost:5173` (Vite dev)
- `http://localhost:3000` (React dev)
- Production domains (add to `.env`)

## 📊 Data Flow

```
Frontend Request
    ↓
FastAPI Endpoint
    ↓
Python Utilities (data_fetcher, indicators, etc.)
    ↓
API Data Sources (yfinance, Alpha Vantage)
    ↓
Response to Frontend
```

## 🚀 Deployment

### Railway.app (Recommended)
1. Push to GitHub
2. Connect GitHub repo to Railway
3. Set environment variables
4. Deploy!

### Render.com
1. Create new Web Service
2. Connect GitHub repo
3. Set build & start commands:
   ```
   Build: pip install -r backend/requirements.txt
   Start: cd backend && python main.py
   ```

### Heroku
```bash
# Login & create app
heroku login
heroku create stocksense-api

# Deploy
git push heroku main

# Set environment variables
heroku config:set ALPHA_VANTAGE_API_KEY=your_key
```

## 🔍 Testing the API

### Using cURL
```bash
# Get stock data
curl http://localhost:8000/api/stock/AAPL?market=US

# Get technical indicators
curl http://localhost:8000/api/indicators/AAPL

# Get prediction
curl http://localhost:8000/api/predict/AAPL?days=30

# Search
curl "http://localhost:8000/api/search?query=apple"
```

### Using Python
```python
import requests

API = "http://localhost:8000"

# Get stock
response = requests.get(f"{API}/api/stock/AAPL", params={"market": "US"})
print(response.json())
```

## 📈 Example Response

```json
{
  "status": "success",
  "symbol": "AAPL",
  "market": "US",
  "data_points": 252,
  "data": [
    {
      "date": "2024-01-10",
      "open": 150.25,
      "high": 152.50,
      "low": 149.80,
      "close": 151.95,
      "volume": 45000000
    },
    ...
  ]
}
```

## 🐛 Troubleshooting

**Port Already in Use?**
```bash
python main.py --port 8001
```

**Module Not Found?**
```bash
pip install -r requirements.txt
```

**CORS Error?**
- Check frontend URL is in `CORS_ORIGINS` in `.env`
- Restart the server after updating

**API Rate Limited?**
- Add delay between requests
- Upgrade API key tier
- Use local caching

## 📚 Documentation

- [FastAPI Docs](https://fastapi.tiangolo.com)
- [yfinance](https://github.com/ranaroussi/yfinance)
- [Alpha Vantage](https://www.alphavantage.co)

---

**Ready to run?** 🚀
```bash
pip install -r requirements.txt
python main.py
```

Then visit `http://localhost:8000/docs` to see interactive API documentation!
