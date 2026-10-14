from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String

from database import Base


class Scan(Base):
    __tablename__ = "scans"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    total_count = Column(Integer, nullable=False, default=0)
