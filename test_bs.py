# Test driver for functions in basic_functions.py

#Import module name as bs
import basic_functions as bs

#test the add 2 functions
#print(bs.add_2(2,3))
#print(bs.add_2(5,6))

#test range
#print(bs.my_range(1,50))
#print(bs.my_range(1, 50, 3))
#print(bs.my_range(1, 50, by=4))
#print(bs.my_range(1,11,3))

#bs.print_triangle(3)
#bs.print_triangle(4)
#bs.print_triangle(25, full=True)

#print(bs.histogram(['a',3,'b','xyz',3,'c','a']))

#print(bs.word_count('.\data\sample_text.txt'))
#print(bs.my_max([11,2,3,4,5,6,1,2,10,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1000000]))

#print(bs.variable_number_of_inputs(2,3))
#print(bs.variable_number_of_inputs(2,3,4,5,6,'Something','Else'))

#print(bs.fzip(lambda x, y: x+y, [1,2,3],[4,5,6]))
#print(bs.fzip(max, [1,2,3],[4,5,6],[7,8,9]))

#print(bs.sum_range(1,10))
#print(bs.sum_range(1,100))

#print(bs.r_rev([1,2,3,'w']))

#print(bs.mfib(1,1,6))
#print(bs.mfib(1,1,10))
#print(bs.mfib(1,1,100))

#print(bs.cartesian_product([1,2],[3,4],[5,6,7]))

print(bs.kcomb([1,2,3,4],2))

#print(map(lambda x: tuple(x), bs.kcomb([1,2,3,4,5,6],3)))

#f1 = lambda x: x+3
#f2 = lambda x: x*x
#f3 = lambda x: x/2.3
#f4 = lambda x: x**0.5

#my_pipe = bs.pipe(f1,f2,f3,f4)
#construct a new function that pipes the above in sequence
#print(map(my_pipe, range(1,21)))
