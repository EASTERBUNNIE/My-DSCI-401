print('Hello World :)')

e1 = ['a','b','c','d','e','f','g']
e2 = [(1,'a'),(2,'b'),(3,'c'),(4,'d'),(5,'e')]

#print out elements in a list of strings
for char in e1:
	print('Next character: ' + char)
print("\n\n")
#print out elements in list of tuples - each field seperatly
for (number, letter) in e2:
	print("The number is: " + str(number))
	print("the letter is: " + str(letter))
	print("-------")

for tuple in e2:
	print("The number is: " + str(tuple[0]))
	print("The letter is: " + str(tuple[1]))
	print("-------")

