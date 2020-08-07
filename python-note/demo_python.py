"""Demo for python"""
import inspect

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

# inspect function
def b1(i):
    if i == 1:
        print('1')
    else:
        print('2')

    return

b1_test = b1(1)

# print(inspect.getsource(b1_test))

a, b, *args = 1,2, 3,4
print('this is', a)
print('this is', b)
print('this is', args)

def func1(a, b, *args):
    print(a)
    print(b)
    print(args)

func1(10,20,'1', '2','ab', 123)

def func2(a, *args):
    return len(args) + 1

print(func2(2,3,4,5))
print(func2(1)) ##to force and ensure that at least an argument given


def func3(*args):
    count = len(args)
    total = sum(args)
    return count and total/count

print(func3(2,2,4,4))
print(func3())

list1 = [12,22,32,42]
func1(*list1) #unpack the list using *args

def func(a, *args, b):
    print(a,args,b)

func(1,'2','3', b=12)

def func(a, *, b):
    print(a,b)

func(1,b=2)

def func(a,b=10, *args):
    print(a,b, args)

func(1,'strign','a')

def func(a, b, **kwargs): #** kwargs will return the dictionary
    print(a,b , kwargs)

func(a = 22,b = 2, c= 1,d=2)

def func(*args, **kwargs):
    print(args, kwargs)

func(1,2,a =1,b=2)



print('Abu bakar')

class Part1:
    x = 0
    def __init__(self, nam):
        self.name = nam
        print('nam', self.name)

    def part_x(self):
        self.x += 1
        print(self.name, self.x)

class Part2(Part1):
    y = 0
    def part_y(self):
        self.y += 4
        # self.x += 5
        self.part_x()
        print('part2', self.name, self.x, self.y)

a = Part1('there')
a.part_x()
a.part_x()
a.part_x()

b = Part1('this')
b.part_x()

c = Part2('inherit')
c.part_y()
c.part_y()

print(12 & 13) #and
print(12|13) #or
print(25&30)
print(12^13) #xor
print(~12) #complement
print(10<<2) #leftshift
print(10>>2) #rightshift




