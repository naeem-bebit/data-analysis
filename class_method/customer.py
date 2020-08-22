class User:
    def log_all(self):
        print("User class")
        print(self)

class Customer(User):

    def __init__(self, name, age):
        self.name = name
        self.age = age

    @property
    def name(self):
        # print('getting name')
        return self._name

    @name.setter
    def name(self, name):
        # print('setting name')
        self._name = name

    def __str__(self):
        return self.name + ' ' + str(self.age)

    def __eq__(self, other):
        if self.name == other.name and self.age == other.age:
            return True
        return False

    def log(self):
        print(self)

    __repr__ = __str__


customer = [Customer('Ali', 22),
            Customer('Abu', 23)]

customer[1].name = 'Aboo'
print(customer[1].name)

print(customer)
print(customer[0] == customer[1])
# customer[0].log()

for i in range(len(customer)):
    customer[i].log()

customer[1].log_all()
