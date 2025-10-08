from sqlalchemy.orm import Session
from models.models import Item
from pydantic import BaseModel

class ItemSchema(BaseModel):
    id: int
    name: str
    description: str

    class Config:
        orm_mode = True

class ItemCreate(BaseModel):
    name: str
    description: str

def get_items(db: Session):
    return db.query(Item).all()

def create_item(db: Session, item: ItemCreate):
    new_item = Item(name=item.name, description=item.description)
    db.add(new_item)
    db.commit()
    db.refresh(new_item)
    return new_item