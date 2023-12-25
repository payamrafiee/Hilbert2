import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv

load_dotenv()

SQLALCHEMY_DATABASE_URL = "postgresql://pr7code:s5rJefhFSHA9@ep-odd-rice-63481895.us-east-2.aws.neon.tech/neondb?sslmode=require"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_session():
    with Session(engine) as session:
        yield session

