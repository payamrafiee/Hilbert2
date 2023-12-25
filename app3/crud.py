from sqlalchemy.orm import Session
from eth_account import Account
from eth_account.messages import encode_defunct
from jose import JWTError, jwt
from datetime import datetime, timedelta
from fastapi import HTTPException, status, Depends, Header
from fastapi.security import OAuth2PasswordBearer

from typing import Dict


from . import models, schemas

SECRET_KEY = "helloworld"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 43200

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()


def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()


def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.User).offset(skip).limit(limit).all()


# def create_user(db: Session, user: schemas.UserCreate):
#     fake_hashed_password = user.password + "notreallyhashed"
#     db_user = models.User(email=user.email, hashed_password=fake_hashed_password)
#     db.add(db_user)
#     db.commit()
#     db.refresh(db_user)
#     return db_user


def get_items(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Item).offset(skip).limit(limit).all()


def create_user_item(db: Session, item: schemas.ItemCreate, user_id: int):
    db_item = models.Item(**item.dict(), owner_id=user_id)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

def get_user_by_public_address(db: Session, public_address: str):
    return db.query(models.User).filter(models.User.public_address == public_address).first()

def create_user(db: Session, user: schemas.UserBase):
    db_user = models.User(**user.dict())
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

async def get_token_header(authorization: str = Header(...)):

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid token")
    return authorization.split(" ")[1]

async def verify_token(token: str = Depends(get_token_header)):
    
    credentials_exception = HTTPException(
        status_code=401, detail="Could not validate credentials"
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise credentials_exception


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def recover_address( data: Dict[str, str]):
  
    address = data["address"]
    signedMessage = data["signedMessage"]
    message = encode_defunct(text=data["message"])
    
    recoveredAddress = Account.recover_message(message, signature=signedMessage)
    return recoveredAddress


