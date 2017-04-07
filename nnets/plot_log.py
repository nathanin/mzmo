#!/usr/bin/python

import argparse
import numpy as np
import re
import os
from matplotlib import pyplot as plt
import time

# Clean up / steal from:
# https://github.com/yassersouri/omgh/blob/master/src/scripts/vis_finetune.py

class_names = ['Test acc', 'Test loss','Train loss']
color_wheel = ['b', 'm', 'r']
test_freq = 1000
loss_freq = 50
def run(logfile, porder, logscale, monitor):
	with open(logfile, 'r') as logfile:
	        log = logfile.read()

	# # Do Iteration pattern first; recycle iterations
	pattern = r"Iteration (?P<iter_num>\d+), loss = (?P<loss_val>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
	train_losses = []
	iterations = []
	for r in re.findall(pattern, log):
		itert = int(r[0])
		loss = float(r[1])

		iterations.append(itert)
		train_losses.append(loss)
	# Now i've got iterations and losses corresponding to the training lines.

	pattern = r"Test net output #0: accuracy = (?P<accuracy>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
	test_accuracy = []
	for r in re.findall(pattern, log):
		x = float(r[0])*100
		test_accuracy.append(x)
	test_accuracy = np.asarray(test_accuracy)

	pattern = r"Test net output #1: loss = (?P<test_loss>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
	test_loss = []
	for r in re.findall(pattern, log):
		x = float(r[0])
		test_loss.append(x)
	test_loss = np.asarray(test_loss)

	iterations = np.asarray(iterations)
	print iterations[-1]
	train_losses = np.asarray(train_losses)
	print "Plotting order {} polynomial fit".format(porder)
	train_losses = np.polyfit(iterations, train_losses, porder)
	train_losses = np.poly1d(train_losses)

	# Set up plot:
	plt.style.use('ggplot')
	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	ax1.set_xlabel('iteration')
	ax1.set_ylabel('loss')
	ax2.set_ylabel('accuracy (%)')
	#ax1.ylim([0.0,1.0])

	# Draw curves
	# Pull out max of iterations

	# http://stackoverflow.com/questions/5484922/secondary-axis-with-twinx-how-to-add-to-legend 

	
	ln0 = ax1.plot(iterations, train_losses(iterations), 
				  color='k',
				  label='Training Loss',
				  linestyle='--')

	# for acc, name, color in zip(class_accuracies, class_names, color_wheel):
	test_iterations = np.linspace(0, iterations[-1], len(test_accuracy))
	test_accuracy = np.polyfit(test_iterations, test_accuracy, porder)
	test_accuracy = np.poly1d(test_accuracy)
	test_loss = np.polyfit(test_iterations, test_loss, porder)
	test_loss = np.poly1d(test_loss)

	ln1 = ax1.plot(test_iterations, #LOSS GOES ON THE LOSS Y-SCALE!! 
			 test_loss(test_iterations), 
			 color=color_wheel[1], 
			 label=class_names[1],
			 linestyle='--')

	ln2 = ax2.plot(test_iterations, 
			 test_accuracy(test_iterations), 
			 color=color_wheel[0], 
			 label=class_names[0])

	lns = ln0 +ln1 +ln2
	labs = [l.get_label() for l in lns]
	plt.legend(lns, labs, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
	           ncol=3, mode="expand", borderaxespad=0.) 

	if logscale:
		print "Plotting iterations in Log scale"
		ax1.set_xscale('log')


# TODO: add a nice way to kill the program... 
# how to combine keyboard entry with a time-out option?
# http://stackoverflow.com/questions/3471461/raw-input-and-timeout
	if monitor:
		print "------------------------------------\n"
		plt.show(block=False) #EZPZ
		time.sleep(60*15) # 15 minute SLEEP. ADD A NICE WAY TO KILL THE PROGRAM IN HERE ~~~
		# Update to create the plot outside this function.. then just keep overwriting it. ?????
		plt.close()
		return True
	else:
		plt.show(block=True)
		return True



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--logfile')
	parser.add_argument('--porder', type=int, default=5) #Low odd number default
	parser.add_argument('--logscale', action="store_true")
	parser.add_argument('--monitor', action="store_true")

	args = parser.parse_args()

	logfile = os.path.expanduser(args.logfile)
	porder = args.porder
	logscale = args.logscale
	monitor = args.monitor

# Possible:
# http://matplotlib.org/examples/user_interfaces/embedding_in_tk.html
# http://stackoverflow.com/questions/11140787/closing-pyplot-windows
	status = True
	if not monitor:
		status = run(logfile, porder, logscale, monitor)
	else:
		while status:
			status = run(logfile, porder, logscale, monitor)





