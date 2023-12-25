from pydantic import BaseModel


class ItemBase(BaseModel):
    title: str
    description: str | None = None


class ItemCreate(ItemBase):
    pass


class Item(ItemBase):
    id: int
    owner_id: int

    class Config:
        orm_mode = True


class UserBase(BaseModel):
    public_address: str


class UserCreate(UserBase):
    password: str


class User(UserBase):
    # items: list[Item] = []
    is_active: bool
    class Config:
        orm_mode = True


class Token(UserBase):
    token: str
    
class SignMessage(BaseModel):
    signedMessage: str
    message: str
    address: str
