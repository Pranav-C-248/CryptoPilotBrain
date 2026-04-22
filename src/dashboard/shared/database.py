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
