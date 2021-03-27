# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     task_db
   Description :
   Author :       hWX889027
   date：          2020/8/5
-------------------------------------------------
"""
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
    __table_args__ = {'extend_existing': True}
    __tablename__ = "task"
    task_id = Column(Integer, primary_key=True)
    task_type = Column(String, server_default='', nullable=False)
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
    def __init__(self):
        self.session = get_session()

    def __del__(self):
        self.session.close()

    def list_tasks(self):
        """

        Returns
        -------

        """
        query = self.session.query(Task)
        tasks = query.all()
        tasks_list = list()
        for task in tasks:
            tasks_list.append(task.to_dict())
        return tasks_list

    def get_task(self, task_id):
        """

        Parameters
        ----------
        task_id

        Returns
        -------

        """
        query = self.session.query(Task).filter_by(task_id=task_id)
        task = query.first()
        return task

    def get_performance(self, task_id):
        """

        Parameters
        ----------
        task_id

        Returns
        -------

        """
        task = self.get_task(task_id)
        if task.performance:
            return json.loads(task.performance)
        else:
            return None

    def get_label(self, task_id):
        """

        Parameters
        ----------
        task_id

        Returns
        -------

        """
        task = self.get_task(task_id)
        if task:
            return task.label
        else:
            return None

    def delete_tasks(self, task_ids):
        """

        Parameters
        ----------
        task_ids

        Returns
        -------

        """
        for task_id in task_ids:
            task = self.get_task(task_id)
            if task:
                dataset = task.dataset
                sample_path = os.path.join(dataset,
                                           "sample_" + str(task_id) + ".csv")
                true_dag_path = os.path.join(dataset,
                                             "true_dag_" + str(task_id) + ".csv")
                node_relationship_path = os.path.join(
                    dataset, "node_relationship_" + str(task_id) + ".csv")
                topo_path = os.path.join(
                    dataset, "topo_" + str(task_id) + ".pkl")
                download_file = os.path.join(dataset, str(task_id) + ".zip")

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

        Parameters
        ----------
        task_type
        task_name
        task_id

        Returns
        -------

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

        Parameters
        ----------
        task_id
        label_path
        evaluation_metrics

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

        Parameters
        ----------
        task_id
        est_dag

        Returns
        -------

        """
        query = self.session.query(Task).filter_by(task_id=task_id).first()
        query.est_dag = json.dumps(est_dag.tolist())
        self.session.commit()
        return True

    def update_true_dag(self, task_id, true_dag):
        """

        Parameters
        ----------
        task_id
        true_dag

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

        Parameters
        ----------
        task_id

        Returns
        -------

        """
        task = self.get_task(task_id)
        if task.est_dag:
            return np.array(json.loads(task.est_dag))
        else:
            return None

    def get_true_dag(self, task_id):
        """

        Parameters
        ----------
        task_id

        Returns
        -------

        """
        task = self.get_task(task_id)
        if task.true_dag:
            return np.array(json.loads(task.true_dag))
        else:
            return None

    def get_task_type(self, task_id):
        """

        Parameters
        ----------
        task_id

        Returns
        -------

        """
        task = self.get_task(task_id)
        if task.task_type:
            return task.task_type
        else:
            return None

    def get_simulation_name(self):
        """

        Returns
        -------

        """
        query = self.session.query(Task).filter_by(task_type="1")
        tasks = query.all()
        name_list = list()
        for task in tasks:
            name_list.append(str(task.task_id) + "_" + task.task_name)
        return name_list

    def update_task_status(self, task_id, status):
        """

        Parameters
        ----------
        task_id
        status

        Returns
        -------

        """
        self.session.query(Task).filter(Task.task_id == task_id).update({
            Task.task_status: status})
        self.session.commit()

    def update_consumed_time(self, task_id, start_time):
        """

        Parameters
        ----------
        task_id
        start_time

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

        Parameters
        ----------
        task_id
        update_time

        Returns
        -------

        """
        query = self.session.query(Task).filter_by(task_id=task_id).first()
        query.update_time = update_time
        self.session.commit()
        return True