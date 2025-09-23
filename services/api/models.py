from sqlalchemy import Column, Integer, BigInteger, Text, JSON, TIMESTAMP, Boolean, Float, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from .db import Base

class Channel(Base):
    __tablename__ = "channel"
    id = Column(Integer, primary_key=True)
    source = Column(Text, nullable=False)
    channel_key = Column(Text, nullable=False)
    units = Column(Text)
    meta = Column(JSON)
    __table_args__ = (UniqueConstraint("source","channel_key", name="uq_source_key"),)

class Telemetry(Base):
    __tablename__ = "telemetry"
    channel_id = Column(Integer, ForeignKey("channel.id"), primary_key=True)
    ts = Column(TIMESTAMP(timezone=True), primary_key=True)
    value = Column(Float, nullable=False)
    quality = Column(Text)

class Anomaly(Base):
    __tablename__ = "anomaly_detection"
    id = Column(BigInteger, primary_key=True)
    channel_id = Column(Integer, ForeignKey("channel.id"), nullable=False)
    window_start = Column(TIMESTAMP(timezone=True), nullable=False)
    window_end   = Column(TIMESTAMP(timezone=True), nullable=False)
    score = Column(Float, nullable=False)
    label = Column(Boolean)
    method = Column(Text, nullable=False)
    params = Column(JSON)
    created_at = Column(TIMESTAMP(timezone=True))
