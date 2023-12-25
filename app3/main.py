from fastapi import Depends, FastAPI, HTTPException, Request, Response, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import Dict
import time

from . import crud, models, schemas
from .database import SessionLocal, engine
import secrets


models.Base.metadata.create_all(bind=engine)

app = FastAPI()

@app.middleware("http")
async def db_session_middleware(request: Request, call_next):
    response = Response("Internal server error", status_code=500)
    try:
        request.state.db = SessionLocal()
        response = await call_next(request)
    finally:
        request.state.db.close()
    return response

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post("/users/", response_model=schemas.User)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    return crud.create_user(db=db, user=user)


@app.get("/users/", response_model=list[schemas.User])
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = crud.get_users(db, skip=skip, limit=limit)
    return users


@app.get("/users/{user_id}", response_model=schemas.User)
def read_user(user_id: int, db: Session = Depends(get_db)):
    db_user = crud.get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user


@app.post("/users/{user_id}/items/", response_model=schemas.Item)
def create_item_for_user(
    user_id: int, item: schemas.ItemCreate, db: Session = Depends(get_db)
):
    return crud.create_user_item(db=db, item=item, user_id=user_id)


@app.get("/items/", response_model=list[schemas.Item])
def read_items(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    items = crud.get_items(db, skip=skip, limit=limit)
    return items


@app.post("/signup/")
def create_user(user: schemas.UserBase, db: Session = Depends(get_db)):
    existing_user = crud.get_user_by_public_address(db, public_address=user.public_address)
    if existing_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User already exists")

    created_user = crud.create_user(db=db, user=user)
    return created_user
    

@app.get('/nonce/')
def generate_nonce():
    # Generate a random 32-byte value to use as the nonce
    nonce = secrets.token_hex(32)
    # Return the nonce value as a JSON object in the response body
    return {"nonce": nonce}


@app.post("/login/")
async def login (data: Dict[str, str], db: Session = Depends(get_db)):
    recoveredAddress = crud.recover_address(data=data)
    if recoveredAddress !=  data['address']:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Signature!")

    token = crud.create_access_token({"address": data["address"]})
    return {"token": token}

@app.post("/verify")
async def verify_jwt(token: dict = Depends(crud.verify_token)):
    current_time = int(time.time())
    
    if token['exp'] < current_time:
        return {"message": "tokenExpired"}
    else:
        return {"message": "ok"}