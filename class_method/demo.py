from time import sleep

print('hello world')

test_c = 6
print(test_c)
print(test_c)
print(test_c)

alist = []
alist.append("B")
alist.append("C")
print(alist)
alist.pop()
print(alist)


class Stack:

    def __init__(self, num1, num2):
        self.items = []
        self.num1 = num1
        self.num2 = num2

    def append_items(self, var):
        return self.items.append(var)

    def pop_items(self):
        return self.pop_items()

    def print_items(self):
        print('This is the list', self.num1 + self.num2, self.items)


print('Class & Method')
s = Stack(1, 2)
s.append_items('A')
s.print_items()

s1 = Stack(3, 4)
s1.print_items()

print("Test function")


def add(x, y):
    return x + y


add(3, 4)


class Student:

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def show(self):
        print(self.name, self.age)

    class Laptop:

        def __init__(self, brand):
            self.brand = brand

        def show(self):
            print(self.brand)


c1 = Student('ali', 16)
c1.show()

print('duck typing')


class Laptop:
    def ide(self, type):
        type.execute()


class Type:
    def execute(self):
        print('Running')


vs = Type()

lap = Laptop()
lap.ide(vs)


class Cat:
    def __init__(self, breed, colour):
        self.breed_1 = breed
        self.colour_1 = colour


cat1 = Cat('kampung', 'kuning')
print(cat1.breed_1)

import numpy as np

list1 = [1, 2, 3, 4, 'a']  # all array must be in the same types

aray_list = np.array(list1)
print(aray_list)
print(type(aray_list))
print(aray_list[1])

a, _, b = (1, 2, 3)  # a = 1, b = 3
print(a, b)

a, *_, b = (7, 6, 5, 4, 3, 2, 1)  # ignore multiple values '_'
print(a, b)

_ = 5  # as variable
while _ < 10:
    print(_, end=' ')  # default value of 'end' id '\n' in python. we're changing it to space
    _ += 1

million = 1_000_000  # separate the digit

print(million)


class Test:
    _a = 1345
    _b = 2345

    def test_this():
        return Test._b


print(Test._a)
print(Test.test_this())


# Single underscores are a Python naming convention indicating a name is meant for internal use.
# It is generally not enforced by the Python interpreter and meant as a hint to the programmer only.

class Stud:

    def __init__(self):
        # self.age = age
        self.b = 'b'
        self.__a = 'magic'


s = Stud()
# print(s.__a)

print(f'{1+2=}')
print(type(f'{1+2=}'))

lista = [1, 2, 3, 4]
aiter = iter(lista)
print(next(aiter))  # == print(iter(aiter).__next__())
print(next(aiter))

listb = np.array([1, 2, 3])
for i in listb:
    print('listb', i)


def a():
    print('a')


a()


class Employee:
    'Common base class for all employees'
    empCount = 0

    def __init__(self, name, salary):
        self.name = name
        self.salary = salary
        Employee.empCount += 1

    def displayCount(self):
        print("Total Employee %d" % Employee.empCount)

    def displayEmployee(self):
        print("Name : ", self.name, ", Salary: ", self.salary)


# The variable empCount is a class variable whose value is shared among all instances of a this class.
# This can be accessed as Employee.empCount from inside the class or outside the class.

e1 = Employee('ravin', 10)
e2 = Employee('Raja', 20)
e3 = Employee('Rajo', 30)

e1.displayEmployee()
e2.displayEmployee()
print("Total Employee %d" % Employee.empCount)

l1 = [1, 2, 3]
print(type(l1))
# print(dir(l1))

print(dir(e3))  # to check the method of the class
bc = 5
bd = 0
try:
    print('Open connection')
    be = bc / bd
    print(be)
except ZeroDivisionError as zero:
    print('Error of 0', zero)
except Exception as e:
    print('This is an error', e)
finally:
    print('Close connection')


# threading
from threading import *
class Hi(Thread):

    def run(self):
        for i in range(5):
            print('Hi')
            sleep(0.2)

class Hello(Thread):

    def run(self):
        for i in range(5):
            print('Hello')
            sleep(0.1)

objHi = Hi()
# objHi.hi()
objHello = Hello()
# objHello.hello()

objHi.start()
objHello.start()

objHi.join()
objHello.join()
print('End of the threading')

# faster than numpy array, numpy array is faster than list,
# its similar concept but different implementation
# numpy array for matrix, multi dimensional array
from array import *
import numpy as np
val1 = array('i', [2,3,4,5]) # i is the type
print(val1)
print(len(val1))
print('this is typecode', val1.typecode)
for i in val1:
    print(i)
val2 = np.array([1,2,3], int)
print('val2', val2)
print('dtype', val2.dtype)
print('type', type(val2))
val2_view = val2.view()

ar1 = np.array([[1,2,3],[4,5,6]])
print(ar1)
m = np.matrix('1 2 3; 4 5 6') # matrix has been deprecated
print('This is matrix', m)
m1 = np.array([[1,2],[3,4]])
m2 = np.array([[1,2],[3,4]])
m3 = m1*m2
print(m3)

print('array matrix: ', ar1.shape)

val2_copy = val2.copy()
val2[1] = 30
print(val2_view)
print(val2_copy)
val2 = array('i',[3,4])
val2.append(5)
print(val2)
ab = [1,2,3,4]
val3 = array('i', (i*2 for i in ab))
print(val3)

print(r'test\navin') #use r for raw string

a = {'a':1, 'b':2, 'c':3} #curly bracket because the value is unique
print(a.get('a'))
print(a['a'])
