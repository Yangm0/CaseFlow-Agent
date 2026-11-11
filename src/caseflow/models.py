from datetime import datetime

from sqlalchemy import DateTime, Index, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class PaymentAttempt(Base):
    __tablename__ = "payment_attempts"
    __table_args__ = (Index("ix_payment_region_time", "region", "attempted_at"),)

    attempt_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    region: Mapped[str] = mapped_column(String(40), nullable=False)
    attempted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    service_name: Mapped[str] = mapped_column(String(80), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)


class ChangeEvent(Base):
    __tablename__ = "change_events"
    __table_args__ = (Index("ix_change_region_time", "region", "occurred_at"),)

    event_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    region: Mapped[str | None] = mapped_column(String(40), nullable=True)
    service_name: Mapped[str] = mapped_column(String(80), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)


class Incident(Base):
    __tablename__ = "incidents"
    __table_args__ = (Index("ix_incident_region_time", "region", "occurred_at"),)

    incident_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    region: Mapped[str | None] = mapped_column(String(40), nullable=True)
    service_name: Mapped[str] = mapped_column(String(80), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False)


class Document(Base):
    __tablename__ = "documents"

    document_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    title: Mapped[str] = mapped_column(String(240), nullable=False)
    source_type: Mapped[str] = mapped_column(String(40), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class DatasetMetadata(Base):
    __tablename__ = "dataset_metadata"

    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[str] = mapped_column(String(500), nullable=False)
