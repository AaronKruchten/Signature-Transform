import numpy as np
from math import factorial
from math import log
import time


# for a given  vector x of arbitrary dimension and depth N
# we compute the final resultant depth
def compute_sig_length(x,N):
	d = len(x)
	numerator = d**(N + 1) - 1
	denominator = d - 1
	return(numerator/denominator)


# this function computes the tensor product of two vectors, x1 and x2
def tensor_product(x1,x2):
	result_length = len(x1)*len(x2)
	result = np.zeros(result_length)
	result_index = 0
	for i in range(len(x1)):
		for j in range(len(x2)):
			result[result_index] = x1[i]*x2[j]
			result_index += 1
	return(result)


# function computes a tensor power of a vector
def tensor_exponent(x,p):
	result = x
	if(p == 0):
		return(np.array([1]))
	for i in range(p - 1):
		result = tensor_product(result,x)
	return(result)

# x is a vector of any length for which we will compute the i'th component of the taylor series of an exponent
# i is the component of which we will compute the taylor series i >= 0
def exp_component(x,i):
	tensor_exponent_value = tensor_exponent(x,i)
	return(tensor_exponent_value/factorial(i))



# x is a vector of any length for which we will compute the exponent
# N is the depth we will compute the result up to
# returns a numpy array of the result
def compute_exp(x,N):
	result_length = int(compute_sig_length(x,N))
	result = np.zeros(result_length)
	curr_index = 0
	for i in range(N+1):
		curr_val = exp_component(x,i)
		curr_val_index = 0
		for j in range(curr_index,curr_index + len(curr_val)):
			result[j] = curr_val[curr_val_index]
			curr_val_index += 1
		curr_index += len(curr_val)
	return(result)


# for two algebras in the tensor algebra, this function computes the product of the two algebras
# X1 = (X1_0,X1_1,X1_2,X1_3,...)
# X1_0 is a vector of length 1
# X1_1 is a vector of length 2
# X1_2 is a vector of length 4
# X1_3 is a vectof of length 8
def tensor_algebra_product(X1,X2,number_of_tensors,d):
	
	result = np.zeros(int(d**(number_of_tensors-1)))
	k = number_of_tensors
	for i in range(k):
		X1_index = i
		X2_index = k - i -1

		X1_lower_index = int(max(0,(d**(X1_index) -1)/(d-1)))
		X1_upper_index = int((d**(X1_index+1)-1)/(d-1))

		X2_lower_index = int(max(0,(d**(X2_index) -1)/(d-1)))
		X2_upper_index = int((d**(X2_index+1)-1)/(d-1)) 

		curr_X1_tensor = X1[X1_lower_index:X1_upper_index]
		curr_X2_tensor = X2[X2_lower_index:X2_upper_index]



		curr_result = tensor_product(curr_X1_tensor,curr_X2_tensor)

		result = result + curr_result
	return result


# for two algebras in the tensor algebra, this function iterates through the algebras getting each series of tensors to compute
# the algebra product
# X1 = (X1_0,X1_1,X1_2,X1_3,...)
# if stream stream is of width 2:
# X1_0 is a vector of length 1
# X1_1 is a vector of length 2
# X1_2 is a vector of length 4
# X1_3 is a vector of length 8
def tensor_algebra_product_iterator(X1,X2,d):
	algebra_length = len(X1)
	result = np.zeros(algebra_length)
	number_of_tensors = int(log(algebra_length - 1,d)) + 1
	for i in range(number_of_tensors):
		tensor_lower_index = 0
		tensor_upper_index = int((d**(i+1) - 1)/(d - 1))
		curr_tensor_X1 = X1[tensor_lower_index:tensor_upper_index]
		curr_tensor_X2 = X2[tensor_lower_index:tensor_upper_index]
		curr_result = tensor_algebra_product(curr_tensor_X1,curr_tensor_X2,i+1,d)
		result_lower_index = int((d**(i) - 1)/(d - 1))
		result_upper_index = int((d**(i+1) - 1)/(d - 1))
		result[result_lower_index:result_upper_index] = curr_result
	return(result)


def compute_full_signature(increment_stream,d,N):
	result = compute_exp(increment_stream[1],N)
	for i in range(2,len(increment_stream)):
		next_part = compute_exp(increment_stream[i],N)
		result = tensor_algebra_product_iterator(result,next_part,d)
	return(result)


# function takes in a stream of data as a numpy nd array
def to_increment_stream(stream,d,n):
	increment_stream = np.zeros((n,d))
	for i in range(1,len(stream)):
		curr_increment = stream[i] - stream[i-1]
		increment_stream[i] = curr_increment
	return(increment_stream)


# takes in a stream which is a numpy ndarray
# N is the depth of the signature transform
def main_compute_signature(stream,N):
	n = len(stream)
	d = len(stream[0])
	increment_stream = to_increment_stream(stream,d,n)
	signature_result = compute_full_signature(increment_stream,d,N)
	return(signature_result)



#################################
# EXAMPLE USAGE
##################################

stream = np.array([
    [1.0, 1.0],
    [3.0, 4.0],
    [5.0, 2.0],
    [8.0, 6.0]

])

depth = 2

sig = main_compute_signature(stream, depth) # compute the signature
print(sig) # prints "[1., 7., 5., 24.5, 19., 16., 12.5]"
