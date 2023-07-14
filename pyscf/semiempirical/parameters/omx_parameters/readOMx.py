import re
import numpy as np
import pandas as pd
from om_readers import *

'''
DATA GSPOM2 ( 6)/      11.4700000D0/
0    1      2 3        4
'''
def read_block_params(filename):
	method_param = []
	parameters = []
	with open(filename,'r') as ofile:
		for line in ofile:
			if re.search('START OF',line):
				method_param.append(parameters)
				parameters = []
			line = line.split()
			leng = len(line)
			if line[0] == 'DATA' and leng > 4:
				parmvar = line[1]
				element = line[3]
				value = line[4]
				parameters.append([parmvar, element, value])
		method_param.append(parameters)
	return method_param[1:]

method_param = read_block_params('BLOCK4.f')

for item in method_param:
	print('elements per method',len(item))

# OM1
om1_reader(method_param)
om2_reader(method_param)
om3_reader(method_param)
odm2_reader(method_param)
odm3_reader(method_param)

