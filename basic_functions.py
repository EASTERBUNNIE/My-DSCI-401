# Some examples of python-style function definitions

def add_2 (x, y):
	return x+y

#illustrate default arguments
def my_range(start, end, by=1):
	thislist = []
	# return range(start, end, by)
	# Homework: rewrite this function to use a for loop rather than resorting to Python's builtin range function
	rng = []
	next = start
	while(next < end):
		rng.append(next)
		next += by
	return rng

# prints a triangle of specified size	
def print_triangle(n, full=False):
	counted = 1
	while(counted <= n):
		print('*' * counted)
		counted+=1
	if full:
		while(counted > 0):
			counted-=1
			print('*' * counted)
	print('\n')
	
def histogram(items):
	#return([[x,items.count(x)] for x in set(items)])
	d = {}
	for i in items:
		if not(d.has_key(i)):
			d[i] = 0
		d[i] += 1
	return d
	
def word_count(file_path, case_sensitive=True, punct = ['!','.',',','"',"'",'?','!']):
	text = open(file_path, 'r').read()
	if not (case_sensitive):
		text = text.lower()
	#text = text.upper()
	#TODO: Add code to count each punctuation character
	#for time being, remove punctuation characters:
	for p in punct:
		text = text.replace(p, ' ')
	words = text.split(' ')
	cleaned_words = []
	for w in words:
		if len(w) > 0:
			cleaned_words.append(w.strip())
	return histogram(cleaned_words)
	
#Returns the maximum(largest) element in the list.	
def my_max(elements):
	if len(elements) > 0:
		curr_max = elements[0]
		for e in elements:
			if e > curr_max:
				curr_max = e
		return curr_max
	return None
	
def variable_number_of_inputs(a, b, *rest):
	print('A is: '+ str(a))
	print('B is: '+ str(b))
	for e in rest:
		print("   Next Optional Input: "+str(e))
		
def fzip(f, *list):
	return map(lambda tup:f(*tup), zip(*list))
	
def sum_range(a, b):
	if a ==b:
		return a
	else:
		return sum_range(a,b-1)+b
		
def r_rev(list):
	if list:
		return r_rev(list[1:])+[list[0]]
	else:
		return []
		
def fib(first, second, n):
	if n==1:
		return first
	if n==2:
		return second
	else:
		return fib(first, second, n-1) + fib(first, second, n-2)

#recursively compute the nth fibonacci number, use memorization to avoid resolving sub problems		
def mfib(first, second, cache):
	if n==1:
		return first
	if n==2:
		return second
	elif cache.has_key(n):
		return cache(n)
	else:
		v = mfib(first, second, n-1) + mfib(first, second, n-2)
		cache[n] = v
		return v
#Compute the cartesian product of a given set	
def cartesian_product(*sets):
	if len(sets) ==1:
		return map(lambda x: [x], sets[0])
	else:
		rest = cartesian_product(*sets[1:])
		combine = lambda x: map(lambda y: [x] + y, rest)
		return reduce(lambda x,y: x+y, map(combine, sets[0]))

#Finds all distinct combinations		
def kcomb(elts, k):
	if len(elts) == k:
		return[elts]
	if k==1:
		return map(lambda x: [x], elts)
	else:
		partials = kcomb(elts[1:], k-1)
		return map(lambda x: [elts[0]] + x, partials) + kcomb(elts[1:], k)
		
#Build a compositional pipe function - build a new function that applies the specified functions in sequence to an input	
def pipe(function_sequence): 
		def applier(input):
			output = input
			for f in function_sequence:
				output = f(output)
			return output
		return applier
		
	
	


		
	
