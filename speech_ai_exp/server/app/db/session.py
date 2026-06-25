from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Base


def get_engine(sql_url: str):
    connect_args = {"check_same_thread": False} if sql_url.startswith("sqlite") else {}
    return create_engine(sql_url, future=True, connect_args=connect_args)


def get_session_factory(sql_url: str) -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(sql_url), autoflush=False, autocommit=False, future=True)


def init_sql_schema(sql_url: str) -> None:
    engine = get_engine(sql_url)
    Base.metadata.create_all(engine)


def reset_sql_schema(sql_url: str) -> None:
    """Drop and recreate all commerce tables (destructive)."""
    engine = get_engine(sql_url)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
