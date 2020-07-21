"""Demo for python"""


# Decorator
def f1():
    # a()
    print('this is function')


def a(f):
    f()

a(f1)
# f1()
# print(f1)


# Recursion

import sys
print(sys.getrecursionlimit()) #the limit is 1000
sys.setrecursionlimit(10)

i = 0
def hello():
    global i # set the global i to 0
    print('Hello', i)
    i = i+ 1
    hello()

# hello() # function calling itself