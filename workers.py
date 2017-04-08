# -*- coding: UTF-8 -*-
'''
Name: Chirag Maheshwari
Course: Search Engine Architecture

Workers
Main Job is to start the workers as 
reducers and mappers and print out their
hostnames on the console.
The servers run on different forked subprocesses
'''
from . import mapper, reducer, inventory

from tornado.ioloop import IOLoop as iol
from tornado import web, process as proc

class WorkerServer(object):
	def __init__(self, port):
		self.__app = None
		self.__port = port

		self._mapped_data = {}

	def start(self):
		if self.__app is None:
			try:
				app = web.Application([
					(r'/reduce', reducer.Reduce),
					(r'/retrieve_reduce_output', reducer.Output),
					(r'/map', mapper.Map, dict(database=self)),
					(r'/retrieve_map_output', mapper.Output, dict(database=self))])
				app.listen(self.__port)
			except Exception as e:
				print(e)
				print("Cannot start server on port: ", self.__port)
				return False

			self.__app = app
		return True

	def add_mapped_data(self, task_id, reducer_idx, data):
		reducer_idx = str(reducer_idx)
		if task_id not in self._mapped_data:
			self._mapped_data[task_id] = {}

		if reducer_idx not in self._mapped_data[task_id]:
			self._mapped_data[task_id][reducer_idx] = []

		self._mapped_data[task_id][reducer_idx].append(data)

	def is_mapped_data(self, task_id, reducer_idx):
		if task_id not in self._mapped_data:
			return False

		reducer_idx = str(reducer_idx)
		if reducer_idx not in self._mapped_data[task_id]:
			return False

		return True

	def get_mapped_data(self, task_id, reducer_idx):
		if not self.is_mapped_data(task_id, reducer_idx):
			return []

		return self._mapped_data[task_id][str(reducer_idx)]

	def delete_mapped_data(self, task_id, reducer_idx):
		if not self.is_mapped_data(task_id, reducer_idx):
			return []

		del self._mapped_data[task_id][str(reducer_idx)]
		if not self._mapped_data[task_id]:
			del self._mapped_data[task_id]

def start_workers():
	pid = proc.fork_processes(inventory.NUM_WORKERS)

	if WorkerServer(inventory.WORKER_PORTS[pid]).start():
		print("Started Worker on port: ", inventory.WORKER_PORTS[pid], "with sub-process-id: ", pid)
	iol.current().start()

if __name__ == "__main__":
	start_workers()