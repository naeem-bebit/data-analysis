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
