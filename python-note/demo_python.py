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

print(sys.getrecursionlimit())  # the limit is 1000
sys.setrecursionlimit(10)

i = 0


def hello():
    global i  # set the global i to 0, it will replace all the value with i =0
    print('Hello', i)
    i = i + 1
    hello()


# hello() # function calling itself


def add(a, *b):  # b will be tuple
    pass
    # c = a+b
    # print(c)


print(add(1, 2))

a = 0
for i in (10, 20, 30):
    # print(i)
    a = i + a
print(a)


def hello_there(*a, **b):  # b is the value. a is the tuple
    print(a)
    print(b)
    for i, j in b.items():
        print(i, j)


hello_there(1, 3, a=3, b=4)

ab = 1234

def some():
    global ab
    print(ab)

    globals()['ab'] = 12 #change the value of the global variable to 12
some()
print(ab)
def some_func(a,b,c):
    print(a,b,c)

# Inheritance

class ab:

    def func1(self):
        print('function 1')

    def func2(self):
        print('function 2')

class abc:

    def func3(self):
        print('function 3')

    def func4(self):
        print('function 4')


class abcd(ab, abc):

    def func5(self):
        print('function 5')

ab_object = abcd()
ab_object.func1()

class a1:
    def __init__(self):
        print('test a1')

a1 = a1()

class a2:

    def __init__(self, val1, val2):
        self.val1 = val1
        self.val2 = val2

    def val_1(self):
        print(self.val1)

    def b2(self):
        print('test a2')

a2 = a2(12,23)
a2.b2()
a2.val_1()