from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from base.config import settings

_engine = None
_Session = None


def _init_engine():
    global _engine, _Session
    if _engine is None:
        _engine = create_engine(
            settings.mysql.url,
            pool_size=10,
            pool_recycle=3600,
            echo=False,
        )
        _Session = sessionmaker(bind=_engine)
    return _Session


def get_mysql_session():
    return _init_engine()()
