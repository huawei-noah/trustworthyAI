# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     main
   Description :
   Author :       hWX889027
   date：          2020/7/23
-------------------------------------------------
"""
import sys
import os
import socket
from flask import Flask
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from web.view.task_view import task
from web.models.models import Base, get_engine
from web.common.config import FILE_PATH


def after_request(resp):
    """

    Parameters
    ----------
    resp

    Returns
    -------

    """
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


def get_host_ip():
    """

    Returns
    -------

    """
    try:
        link = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        link.connect(('8.8.8.8', 80))
        host_ip = link.getsockname()[0]
    finally:
        link.close()
    return host_ip


app = Flask(__name__, static_url_path="")
app.after_request(after_request)
app.register_blueprint(task, url_prefix='/task')

if __name__ == '__main__':
    if not os.path.exists(FILE_PATH):
        os.makedirs(FILE_PATH)
    Base.metadata.create_all(get_engine(), checkfirst=True)
    app.run(host=get_host_ip(), port=5000)
