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

import json
import os
import shutil
import datetime
from sqlalchemy import Column, String, Integer, Text, DateTime
from sqlite3 import DatabaseError
import numpy as np

from web.models.models import Base, get_session
from web.common.config import FILE_PATH


class Task(Base):
    """
    Mapping class from ORM schema to database task table.

    Parameters
    ----------
    task_id: int
        task key in the database.
    task_type: int
        The value 1 indicates that the task is a data generation task.
        The value 2 indicates that the task is a causal discovery task.
    task_name: str
        Indicates the task name entered by the user. The user can enter the task name freely.
    dataset: str
        When the task type is 1, it is a directory for storing the data files to be generated.
        When the task type is 2, the value is a file path or a built-in data file name.
    algorithm: str
        Operator name of the current task.
    create_time: datetime.datetime
        Task creation time.
    update_time: datetime.datetime
        Time when the task status is updated.
    consumed_time: str
        Task Execution Duration.
    task_status: str
        Task completion percentage. A non-numeric value indicates failure.
    performance:
        Algorithm evaluation indicators and corresponding values.
    label: str
        Built-in data name or real image file name.
    est_dag:
        Predictive graphs for causal discovery.
    true_dag:
        Realistic graph of data used for causal discovery.
    """
    __table_args__ = {'extend_existing': True}
    __tablename__ = "task"
    task_id = Column(Integer, primary_key=True)
    task_type = Column(Integer)
    task_name = Column(String, nullable=False)
    dataset = Column(Text, server_default='', nullable=False)
    algorithm = Column(String, server_default='', nullable=False)
    parameters = Column(Text, server_default='', nullable=False)
    create_time = Column(DateTime, default=datetime.datetime.now(),
                         nullable=False)
    update_time = Column(DateTime, default=datetime.datetime.now(),
                         nullable=False)
    consumed_time = Column(String)
    task_status = Column(String)
    performance = Column(Text)
    label = Column(String)
    est_dag = Column(Text)
    true_dag = Column(Text)

    def to_dict(self):
        """
        Obtains the key-value pair information of an attribute field.

        Returns
        -------
        {}
        """
        return {"task_id": self.task_id,
                "task_type": self.task_type,
                "task_name": self.task_name,
                "file_name": self.dataset,
                "create_time": self.create_time.strftime("%Y-%m-%d %H:%M:%S"),
                "update_time": self.update_time.strftime("%Y-%m-%d %H:%M:%S"),
                "consumed_time": self.consumed_time,
                "task_status": self.task_status,
                "performance": json.loads(self.performance) if self.performance
                else {},
                "label": self.label}


class TaskApi(object):
    """
    Interface for accessing task-related fields in the database.

    Attributes
    ----------
    session: sqlalchemy.orm.session.Session.
        Session Instance.
    """
    def __init__(self):
        self.session = get_session()

    def __del__(self):
        self.session.close()

    def list_tasks(self):
        """
        Obtaining Task List Information.

        Returns
        -------
        tasks_list: list
            task list
        """
        query = self.session.query(Task)
        tasks = query.all()
        tasks_list = list()
        for task in tasks:
            tasks_list.append(task.to_dict())
        return tasks_list

    def get_task(self, task_id):
        """
        Searching for Tasks by Task ID.

        Parameters
        ----------
        task_id: int
            task key in the database.

        Returns
        -------
        task: dict
            Task Fields.
        """
        query = self.session.query(Task).filter_by(task_id=task_id)
        task = query.first()
        return task

    def get_performance(self, task_id):
        """
        Obtains performance field information based on task ID.

        Parameters
        ----------
        task_id: int
            task key in the database.

        Returns
        -------
        : dict
        If there is performance information, the performance information dictionary is returned. Otherwise, None is returned.
        """
        task = self.get_task(task_id)
        if task.performance:
            return json.loads(task.performance)
        else:
            return None

    def get_label(self, task_id):
        """
        Obtains built-in data name or real image file name.

        Parameters
        ----------
        task_id: int
            task key in the database.
        Returns
        -------
        :str
            Realistic Map Fields.
        """
        task = self.get_task(task_id)
        if task:
            return task.label
        else:
            return None

    def delete_tasks(self, task_ids):
        """
        Delete tasks based on the task IDs list.

        Parameters
        ----------
        task_ids: list
            task id list.
        Returns
        -------

        """
        for task_id in task_ids:
            task = self.get_task(task_id)
            if task:
                data = task.dataset
                task_name = self.get_task_name(task_id)
                sample_path = os.path.join(data, "datasets", str(task_id) + "_" + task_name + ".csv")
                true_dag_path = os.path.join(data, "true", str(task_id) + "_" + task_name + ".npz")
                node_relationship_path = os.path.join(data, "node_relationship_" + str(task_id) + "_" + task_name + ".csv")
                topo_path = os.path.join(data, "topo_" + str(task_id) + "_" + task_name + ".npz")
                download_file = os.path.join(FILE_PATH, str(task_id) + ".zip")

                if os.path.exists(sample_path):
                    os.remove(sample_path)
                if os.path.exists(true_dag_path):
                    os.remove(true_dag_path)
                if os.path.exists(node_relationship_path):
                    os.remove(node_relationship_path)
                if os.path.exists(topo_path):
                    os.remove(topo_path)
                if os.path.exists(download_file):
                    os.remove(download_file)

                result_data = os.path.join(FILE_PATH, "task", str(task_id))
                if os.path.exists(result_data):
                    shutil.rmtree(result_data)

                download_file = os.path.join(FILE_PATH, "task", str(task_id) + ".zip")
                if os.path.exists(download_file):
                    os.remove(download_file)

                self.session.delete(task)
                self.session.commit()
        return True

    def add_task(self, task_type, task_name, task_id=None):
        """
        Create or update a task.

        Parameters
        ----------
        task_type: int
            1 or 2.
        task_name: str
            The user enters a character string.
        task_id: int
            task key in the database.
        Returns
        -------
        task_id: int
            task key in the database.
        """
        try:
            if task_id is None:
                create_time = datetime.datetime.now()
                task = Task(task_type=task_type, task_name=task_name, create_time=create_time)
                self.session.add(task)
                self.session.commit()
                task_id = task.task_id
            else:
                self.session.query(Task).filter(
                    Task.task_id == task_id).update({
                    Task.task_name: task_name
                })
                self.session.commit()
        except DatabaseError:
            return None

        return task_id

    def update_performance(self, task_id, label_path, evaluation_metrics):
        """
        Update Performance Fields.

        Parameters
        ----------
        task_id: int
            task key in the database.
        label_path: str
            Realistic path string or built-in data name.
        evaluation_metrics: dict
            Evaluation indicator key-value pair.
        Returns
        -------

        """
        query = self.session.query(Task).filter_by(task_id=task_id).first()
        query.label = label_path
        query.performance = json.dumps(evaluation_metrics)
        self.session.commit()
        return True

    def update_est_dag(self, task_id, est_dag):
        """
        Update Prediction Chart.

        Parameters
        ----------
        task_id: int
            task key in the database.
        est_dag: str
            Prediction Graph String.
        Returns
        -------

        """
        query = self.session.query(Task).filter_by(task_id=task_id).first()
        query.est_dag = json.dumps(est_dag.tolist())
        self.session.commit()
        return True

    def update_true_dag(self, task_id, true_dag):
        """
        Update Realistic Graph String.

        Parameters
        ----------
        task_id: int
            task key in the database.
        true_dag: str
            Realistic Graph String.
        Returns
        -------

        """
        if true_dag is None:
            return False
        query = self.session.query(Task).filter_by(task_id=task_id).first()
        query.true_dag = json.dumps(true_dag.tolist())
        self.session.commit()
        return True

    def get_est_dag(self, task_id):
        """
        Obtains  Prediction Chart.
        Parameters
        ----------
        task_id: int
            task key in the database.

        Returns
        -------
        :np.array or None

        """
        task = self.get_task(task_id)
        if task.est_dag:
            return np.array(json.loads(task.est_dag))
        else:
            return None

    def get_true_dag(self, task_id):
        """
        Obtains  Realistic Graph String.
        Parameters
        ----------
        task_id: int
            task key in the database.

        Returns
        -------
        :np.array or None

        """
        task = self.get_task(task_id)
        if task.true_dag:
            return np.array(json.loads(task.true_dag))
        else:
            return None

    def get_task_type(self, task_id):
        """
        Obtains task type.

        Parameters
        ----------
        task_id: int
            task key in the database.

        Returns
        -------
        :int
            1 or 2.
        """
        task = self.get_task(task_id)
        if task.task_type:
            return task.task_type
        else:
            return None

    def get_task_name(self, task_id):
        """
        Obtains task name.

        Parameters
        ----------
        task_id: int
            task key in the database.

        Returns
        -------
        : str
        """
        task = self.get_task(task_id)
        if task.task_name:
            return task.task_name
        else:
            return None

    def get_simulation_name(self):
        """
        Obtains the built-in data name list.
        Returns
        -------
        name_list: list
            Data generated by a data generation task in the task list.
        """
        query = self.session.query(Task).filter_by(task_type="1")
        tasks = query.all()
        name_list = list()
        for task in tasks:
            name_list.append(str(task.task_id) + "_" + task.task_name)
        return name_list

    def update_task_status(self, task_id, status):
        """
        Update task status.

        Parameters
        ----------
        task_id: int
            task key in the database.
        status: str
            task status.

        Returns
        -------

        """
        self.session.query(Task).filter(Task.task_id == task_id).update({
            Task.task_status: status})
        self.session.commit()

    def update_consumed_time(self, task_id, start_time):
        """
        Update consumed time.

        Parameters
        ----------
        task_id: int
            task key in the database.
        start_time: datetime.datetime
            Task Start Time.

        Returns
        -------

        """
        end_time = datetime.datetime.now()
        consumed_time = end_time - start_time
        self.session.query(Task).filter(Task.task_id == task_id).update({
            Task.consumed_time: round(consumed_time.total_seconds(),2)})
        self.session.commit()

    def update_update_time(self, task_id, update_time):
        """
        Update task update time.

        Parameters
        ----------
        task_id: int
            task key in the database.
        update_time: datetime.datetime
            Task update time.
        Returns
        -------

        """
        query = self.session.query(Task).filter_by(task_id=task_id).first()
        query.update_time = update_time
        self.session.commit()
        return True