import os
from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class AuditLog(Base):
    __tablename__ = 'audit_logs'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, index=True, default=lambda: datetime.now(timezone.utc))
    symbol = Column(String, nullable=False)
    action = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    factors = Column(JSON, nullable=True)
    explanation = Column(Text, nullable=True)
    outcome = Column(String, nullable=True)
    profit_loss = Column(Float, nullable=True)

class CachedNews(Base):
    __tablename__ = 'cached_news'

    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    url = Column(String, nullable=False)
    source = Column(String, nullable=False)
    published = Column(DateTime, nullable=False)
    sentiment_label = Column(String, nullable=True)
    sentiment_score = Column(Float, nullable=True)
    symbol = Column(String, nullable=True)
    cached_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

class Portfolio(Base):
    __tablename__ = 'portfolio'

    id = Column(Integer, primary_key=True)
    balance = Column(Float, nullable=False, default=10000.0) # Default $10k starting balance for paper trading
    updated_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

class Position(Base):
    __tablename__ = 'positions'

    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False, unique=True)
    quantity = Column(Float, nullable=False, default=0.0)
    average_price = Column(Float, nullable=False, default=0.0)
    updated_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

class TradeLedger(Base):
    __tablename__ = 'trade_ledger'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, index=True, default=lambda: datetime.now(timezone.utc))
    symbol = Column(String, nullable=False)
    action = Column(String, nullable=False) # 'BUY' or 'SELL'
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    fee = Column(Float, nullable=False, default=0.0)
    realized_pnl = Column(Float, nullable=True) # Null for BUYs, calculated for SELLs

class QueuedTrade(Base):
    __tablename__ = 'queued_trades'

    id = Column(Integer, primary_key=True)
    asset_name = Column(String, nullable=False)
    signal = Column(String, nullable=False) # 'BUY' or 'SELL'
    entry_condition = Column(String, nullable=False)
    exit_condition = Column(String, nullable=False)
    target = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=False)
    position_size = Column(Float, nullable=False)
    timeframe = Column(String, nullable=False)
    
    status = Column(String, nullable=False, default="UNENTERED") # UNENTERED, ENTERED, CLOSED
    
    # Populated when entering
    entry_price = Column(Float, nullable=True)
    entry_time = Column(DateTime, nullable=True)
    
    # Populated when exiting
    exit_price = Column(Float, nullable=True)
    exit_time = Column(DateTime, nullable=True)
    realized_pnl = Column(Float, nullable=True)

# Ensure data directory exists
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'data')
os.makedirs(data_dir, exist_ok=True)
db_path = os.path.join(data_dir, 'cryptopilot.db')

engine = create_engine(f'sqlite:///{db_path}', echo=False)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
