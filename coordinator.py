# -*- coding: UTF-8 -*-
'''
Coordinator
Coordinates between the mappers and reducers
'''
import argparse, os, urllib, json
from . import inventory
from tornado import httpclient as httpc, gen
from tornado.ioloop import IOLoop

parser = argparse.ArgumentParser(prog="Map-Reduce Framework")

parser.add_argument("--mapper_path", help="Mapper program to run", type=str, default=None)
parser.add_argument("--reducer_path", help="Reducer program to run", type=str, default=None)
parser.add_argument("--job_path", help="Job path", type=str, default=None)
parser.add_argument("--num_reducers", help="Number of reducers to use", type=int, default=1)
parser.add_argument("--timeout", help="Timeout for any request", type=float, default=None)

opt = parser.parse_args()

@gen.coroutine
def map_requests():
	futures = []
	i = 0
	client = httpc.AsyncHTTPClient()
	for filename in os.listdir(opt.job_path):
		if filename.endswith(".in"):
			server = inventory.WORKER_HOSTS[i % inventory.NUM_WORKERS]
			params = urllib.parse.urlencode({
				'mapper_path': opt.mapper_path,
				'input_file': os.path.join(opt.job_path, filename),
				'num_reducers': opt.num_reducers})
			url = "http://{}/map?{}".format(
				server, params)
			futures.append(client.fetch(url, request_timeout=opt.timeout))
			i += 1
	responses = yield futures

	return responses

@gen.coroutine
def reduce_requests(map_ids):
	futures = []
	client = httpc.AsyncHTTPClient()
	for i in range(opt.num_reducers):
		server = inventory.WORKER_HOSTS[i % inventory.NUM_WORKERS]
		params = urllib.parse.urlencode({
				'reducer_ix': i,
				'reducer_path': opt.reducer_path,
				'map_task_ids': ",".join(map_ids),
				'job_path': opt.job_path})
		url = "http://{}/reduce?{}".format(
			server, params)
		futures.append(client.fetch(url, request_timeout=opt.timeout))
	responses = yield futures

	return responses

@gen.coroutine
def main():
	print("Starting mappers...")
	mappers = yield map_requests()

	print("Mappers completed their task")
	map_task_ids = []
	for map_response in mappers:
		# print(map_response.request_time)
		response = json.loads(map_response.body.decode())
		if response['status'] == 'success':
			map_task_ids.append(response['map_task_id'])
		else:
			print("Some error in response from mapper. Exiting....")
			exit()

	print("Starting reducers...")
	# print(map_task_ids)
	reducers = yield reduce_requests(map_task_ids)

	print("Reducers completed their task...")
	for reduce_response in reducers:
		response = json.loads(reduce_response.body.decode())
		if response['status'] != 'success':
			print("Some error in response from reducer. Exiting....")
			exit()

IOLoop.current().run_sync(main)