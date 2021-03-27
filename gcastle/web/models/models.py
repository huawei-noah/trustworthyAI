# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     model
   Description :
   Author :       hWX889027
   date：          2020/7/27
-------------------------------------------------
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


def get_engine():
    engine = create_engine("sqlite:///data/task.db?check_same_thread=False")
    return engine


def get_session_maker(engine):
    session_maker = sessionmaker(bind=engine)
    return session_maker


def get_session():
    engine = get_engine()
    maker = get_session_maker(engine)
    session = maker()
    return session
