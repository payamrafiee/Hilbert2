from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "postgresql://pr7code:6A1PxwNREOkU@ep-lingering-glade-79913960.us-east-2.aws.neon.tech/neondb?sslmode=require"
# SQLALCHEMY_DATABASE_URL = "postgresql://user:password@postgresserver/db"

metadata = MetaData()

engine = create_engine(
    SQLALCHEMY_DATABASE_URL
)

def drop_tables():
    metadata.reflect(bind=engine)
    
    # Drop all tables
    metadata.drop_all(bind=engine, checkfirst=True)

    # Recreate all tables
    metadata.create_all(bind=engine)

# Call the function to drop and recreate tables
# drop_tables()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
