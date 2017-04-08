# -*- coding: UTF-8 -*-
'''
Inventory
Constants for different components
'''
import hashlib, getpass

NUM_WORKERS = 4

MAX_PORT = 50000
MIN_PORT = 10000
BASE_PORT = int(hashlib.md5(getpass.getuser().encode()).hexdigest()[:8], 16) % \
	(MAX_PORT - MIN_PORT) + MIN_PORT

WORKER_PORTS = [port for port in range(BASE_PORT, BASE_PORT + NUM_WORKERS)]
WORKER_HOSTS = ['127.0.0.1:{}'.format(port) \
	for port in WORKER_PORTS]
