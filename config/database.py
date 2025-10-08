from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from config.settings import settings

# print(settings.DATABASE_URL)
# Configure connection pool with pre-ping to verify connections
try:
    engine = create_engine(
        settings.DATABASE_URL,
        pool_pre_ping=True,  # Verify connections before using
        pool_size=5,
        max_overflow=10,
        pool_recycle=300,  # Recycle connections after 5 minutes
        echo_pool=False,
        connect_args={
            "connect_timeout": 10,
            "options": "-c statement_timeout=30000",  # 30 second statement timeout
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5
        }
    )
except Exception as e:
    print(f"Error creating database engine: {e}")
    # Fallback to NullPool if connection args fail
    engine = create_engine(
        settings.DATABASE_URL,
        poolclass=NullPool,
        pool_pre_ping=True
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """Get a fresh database session with proper cleanup."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
