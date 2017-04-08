# -*- coding: UTF-8 -*-
'''
Mappers
Handles the map request
'''
from tornado import web, gen
import json, subprocess, io, pickle, hashlib, random

class Map(web.RequestHandler):
	def initialize(self, database):
		self.__database = database

	def _fetch_arguments(self):
		self._mapper_path = self.get_query_argument(
			"mapper_path", 
			strip=False)

		self._input_file = self.get_query_argument(
			"input_file", 
			strip=False)

		self._num_reducers = int(self.get_query_argument(
			"num_reducers",
			default="1",
			strip=False))

	def _get_unique_id(self):
		return hashlib.md5(str(random.random()).encode()).hexdigest()

	def _get_reducer_index(self, key):
		if type(key) is not int:
			key = int(hashlib.md5((str(key)).encode()).hexdigest()[:10], 16)
		return key % self._num_reducers

	def _store_data(self, data):
		stream = io.BytesIO(data)
		while True:
			try:
				key, val = pickle.load(stream)
				reducer_index = self._get_reducer_index(key)
				self.__database.add_mapped_data(
						self._task_id, 
						reducer_index,
						(key, val))
			except EOFError:
				break


	def _emit_data(self):
		self._status = False

		p = subprocess.Popen(self._mapper_path, 
			stdin=subprocess.PIPE, stdout=subprocess.PIPE)
		(out, _) = p.communicate(open(self._input_file, "rb").read())

		self._task_id = self._get_unique_id()
		self._store_data(out)

		self._status = True

	def _get_response(self):
		if self._status:
			return {
				'status': 'success',
				'map_task_id': str(self._task_id)
			}
		else:
			return {
				'status': 'failure'
			}

	@gen.coroutine
	def get(self):
		self._fetch_arguments()
		self._emit_data()
		response = self._get_response()
		self.write(response)

class Output(web.RequestHandler):
	def initialize(self, database):
		self.__database = database

	def _fetch_arguments(self):
		self._reducer_idx = int(self.get_query_argument(
			"reducer_ix", 
			default="0", 
			strip=False))
		self._map_task_id = self.get_query_argument(
			"map_task_id",
			default="",
			strip=False)

	def _get_response(self):
		if self._map_task_id:
			toReturn = json.dumps(self.__database.get_mapped_data(
				self._map_task_id,
				self._reducer_idx))
			self.__database.delete_mapped_data(
				self._map_task_id,
				self._reducer_idx)

		return toReturn

	@gen.coroutine
	def get(self):
		self._fetch_arguments()
		response = self._get_response()
		self.write(response)

if __name__ == "__main__":
    pass