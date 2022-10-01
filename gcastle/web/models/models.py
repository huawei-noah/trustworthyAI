# coding=utf-8
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


def get_engine():
    """
    Create a database connection engine.

    Returns
    -------
    engine: sqlalchemy.engine.base.Engine
        Database Access Engine.
    """
    engine = create_engine("sqlite:///data/task.db?check_same_thread=False")
    return engine


def get_session_maker(engine):
    """
    Creating a Session class.

    Parameters
    ----------
    engine: sqlalchemy.engine.base.Engine
        Database Access Engine.

    Returns
    -------
    session_maker: sqlalchemy.orm.session.sessionmaker
        Session class.
    """
    session_maker = sessionmaker(bind=engine)
    return session_maker


def get_session():
    """
    Creating a Session Instance.

    Returns
    -------
    session: sqlalchemy.orm.session.Session.
        Session Instance.
    """
    engine = get_engine()
    maker = get_session_maker(engine)
    session = maker()
    return session
