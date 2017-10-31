#Geoffry Berryman DSCI 401 Homework 1

def flatten(y):
  answer = []
  for item in y:
    if type(item) == type([]):
      answer.extend(flatten(item))
    else:
	  answer.append(item)
  return answer
  
def powerset(z):
	the_set = [[]]
	count = 1
	for item in z:
		for thing in the_set:
			the_set = the_set + [list(thing)+[item]]
			count = count+1
	print ("Number of combinations: ",count)
	return the_set


def all_perms(z):
    if not z:
            return [[]]
    answer = []
    for thing in z:
            temp = z[:]
            temp.remove(thing)
            answer.extend([[thing] + a for a in all_perms(temp)])
    return answer
	
		
def  numspi(num, d):
	max = num*num
	size = num
	nums = list(range(0,max))
	count = 0
	my_list = []
	for item in nums:
		my_list.append(item)
	while(count < num):
		for item in my_list:
			my_list = my_list.extend(numspi(item,d))
	print('I have no clue what Im doing :) ')	
	return(my_list)	
	