import plotly
import plotly.graph_objs as go 

import numpy as np 
import pickle
import os 

data_type = 'space_shuttle' 
data = dict() 

for filename in os.listdir(data_type):
	if not filename.endswith('.txt'): 
		continue 

	data[filename] = [] 
	
	with open(os.path.join(data_type, filename), 'r') as f: 
		for line in f:
			tokens = line.split()
			data[filename].append(tokens) 


for filename, value in data.items(): 
	value = list(zip(*value))
	graph1 = go.Scatter( 
		x = list(range(len(value[0]))), 
		y = value[0], 
		mode = 'lines + markers', 
		name = filename, 
		line = dict(width=1), 
	) 
	graph2 = go.Scatter( 
		x = list(range(len(value[0]))), 
		y = [0]*len(value[0]), 
		mode = 'lines + markers', 
		name = 'Anomaly',  
		line = dict(width=1), 
	) 


	plotly.offline.plot({'data':[graph1, graph2], 'layout': go.Layout(title=filename,)}, filename=os.path.join('display_'+data_type, filename+'.html')) 


