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

import torch
import sys
import os
import socket
from flask import Flask, request
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from web.view.task_view import task
from web.models.models import Base, get_engine
from web.common.config import lang_str
from web.common.utils import set_current_language, update_inline_datasets


def before_first_request():
    """
	Set Language
    Returns
    -------

    """
    lang = request.cookies.get('lang')
    if lang is None:
        lang = 'zh'
    set_current_language(lang_str.get(lang))


def after_request(resp):
    """
    Access Cross-Domain Access.

    Parameters
    ----------
    resp: flask.wrappers.Response
        set *
    Returns
    -------
    resp: flask.wrappers.Response
    """
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


def get_host_ip():
    """
    Obtain the IP address of the local host.

    Returns
    -------
    host_ip: str
        ip str
    """
    try:
        link = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        link.connect(('8.8.8.8', 80))
        host_ip = link.getsockname()[0]
    finally:
        link.close()
    return host_ip


app = Flask(__name__, static_url_path="")
app.before_first_request(before_first_request)
app.after_request(after_request)
app.register_blueprint(task, url_prefix='/task')

if __name__ == '__main__':
    update_inline_datasets()
    Base.metadata.create_all(get_engine(), checkfirst=True)
    app.run(host=get_host_ip(), port=5000)
