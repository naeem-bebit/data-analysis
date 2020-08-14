class A:
    def __init__(self):
        self.b = B(self)
        print(self, self.b)

class B:
    def __init__(self, a):
        self.a = a
        print(self, self.a)

my_var = A()