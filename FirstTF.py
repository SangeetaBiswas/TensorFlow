#!/home/sangeeta/Tensorflow/bin/python

#	Sangeeta Biswas 27.6.2018
#	------------------------------------------------------
#	1.	General convention is to use 'tf' as nickname
#		of tensorflow.
#	2.	TensoFlow program takes longer time than a Python 
#		program.
#	3.	Normal assignment statements are considered as 
#		defining an operation in a graph. As long as we 
#		do not run that operation under a session, we 
#		will not get our desired output. Instead we will
#		get a structure of a tensor.
#		Therefore, the output of the first print() is
#		Tensor("Const:0", shape=(), dtype=string)
#		The second print() gives our desired output
#		Hello, TensorFlow!
#	4.	It is a good practice to release resources 
#		occupied in a session at the end of session by
#		calling close().
#		e.g., sess.close()
#	------------------------------------------------------

import tensorflow as tf

msg = tf.constant('Hello, TensorFlow!')
print(msg)
sess = tf.Session()
print(sess.run(msg))
sess.close()
