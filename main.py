"""
FastAPI Backend for StockSense
Main application entry point
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta

from utils.data_fetcher import get_stock_data, get_company_overview

app = FastAPI(
    title="StockSense API",
    description="Professional stock market analysis API",
    version="1.0.0"
)

# CORS middleware - allow frontend to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# HEALTH CHECK
# ============================================

@app.get("/")
def read_root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "StockSense API is running",
        "version": "1.0.0"
    }

# ============================================
# STOCK DATA ENDPOINTS
# ============================================

@app.get("/api/stock/{symbol}")
def get_stock(
    symbol: str,
    days: int = 365
):
    """
    Get stock data for a given symbol
    
    Args:
        symbol: Stock symbol (e.g., AAPL, RELIANCE.NS)
        days: Number of days of historical data
    
    Returns:
        Stock data with OHLCV
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        data = get_stock_data(
            symbol=symbol,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )
        
        if data is None or data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        # Convert to list of dicts for JSON serialization
        records = []
        for idx, row in data.iterrows():
            records.append({
                "date": str(idx.date()) if hasattr(idx, 'date') else str(idx),
                "open": float(row.get('Open', 0)),
                "high": float(row.get('High', 0)),
                "low": float(row.get('Low', 0)),
                "close": float(row.get('Close', 0)),
                "volume": int(row.get('Volume', 0))
            })
        
        return {
            "status": "success",
            "symbol": symbol,
            "data_points": len(records),
            "data": records
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# TECHNICAL INDICATORS ENDPOINTS
# ============================================

@app.get("/api/indicators/{symbol}")
def get_indicators(
    symbol: str,
    days: int = 365
):
    """
    Get technical indicators for a stock
    (Coming soon - requires additional setup)
    """
    return {
        "status": "coming_soon",
        "symbol": symbol,
        "message": "Technical indicators endpoint coming soon"
    }

# ============================================
# PREDICTION ENDPOINTS
# ============================================

@app.get("/api/predict/{symbol}")
def get_prediction(
    symbol: str,
    days: int = 30
):
    """
    Get price prediction for a stock
    (Coming soon - requires additional setup)
    """
    return {
        "status": "coming_soon",
        "symbol": symbol,
        "message": "Price prediction endpoint coming soon"
    }

# ============================================
# COMPANY INFO ENDPOINTS
# ============================================

@app.get("/api/company/{symbol}")
def get_company(symbol: str):
    """
    Get company information
    
    Args:
        symbol: Stock symbol
    
    Returns:
        Company details
    """
    try:
        info = get_company_overview(symbol)
        return {
            "status": "success",
            "symbol": symbol,
            "info": info if info else {}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# POPULAR STOCKS ENDPOINTS
# ============================================

@app.get("/api/popular-stocks")
def get_popular_stocks():
    """
    Get list of popular stocks for quick access
    
    Returns:
        Popular global and Indian stocks
    """
    popular = {
        "global": [
            {"symbol": "AAPL", "name": "Apple", "market": "US"},
            {"symbol": "MSFT", "name": "Microsoft", "market": "US"},
            {"symbol": "GOOGL", "name": "Google", "market": "US"},
            {"symbol": "AMZN", "name": "Amazon", "market": "US"},
            {"symbol": "TSLA", "name": "Tesla", "market": "US"},
        ],
        "indian": [
            {"symbol": "RELIANCE.NS", "name": "Reliance Industries", "market": "NSE"},
            {"symbol": "TCS.NS", "name": "Tata Consultancy Services", "market": "NSE"},
            {"symbol": "INFY.NS", "name": "Infosys", "market": "NSE"},
            {"symbol": "HDFC.NS", "name": "HDFC Bank", "market": "NSE"},
            {"symbol": "WIPRO.NS", "name": "Wipro", "market": "NSE"},
        ]
    }
    
    return {
        "status": "success",
        "stocks": popular
    }

# ============================================
# SEARCH ENDPOINT
# ============================================

@app.get("/api/search")
def search_stocks(query: str):
    """
    Search for stocks by symbol or name
    
    Args:
        query: Search query
    
    Returns:
        List of matching stocks
    """
    # This would query a database in production
    # For now, return popular stocks that match
    
    all_stocks = [
        ("AAPL", "Apple"),
        ("MSFT", "Microsoft"),
        ("GOOGL", "Google"),
        ("AMZN", "Amazon"),
        ("TSLA", "Tesla"),
        ("RELIANCE.NS", "Reliance Industries"),
        ("TCS.NS", "Tata Consultancy Services"),
        ("INFY.NS", "Infosys"),
    ]
    
    query = query.upper()
    results = [
        {"symbol": symbol, "name": name}
        for symbol, name in all_stocks
        if query in symbol or query in name.upper()
    ]
    
    return {
        "status": "success",
        "query": query,
        "results": results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
