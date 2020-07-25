class Student:
    def __init__(self, name, age, *args):
        self.name_first = name
        self.age_first = age
        self.mark_1 = args

    def show(self):
        print(self.name_first, self.age_first, self.mark_1)

    def total_mark(self, mark2):
        return self.age_first + mark2
