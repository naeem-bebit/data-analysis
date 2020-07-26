from Student import Student
from Student import Teacher

p1 = Student('ali',3, 10)
print(p1.name_first)


p2 = Student('abu',4, 20)
print(p2.name_first)

p1.show()
p2.show()

print(p1.total_mark(2))
print(p2.total_mark(2))

p3 = Student('Ais', 5, 'a',1, 'b')
p3.show()
print(p3.mark_1)

t1 = Teacher('Cikgu ali')
t1.show_teacher()

t1 = Student('abubakar', 45, 90)
t1.show()

# Operator overloading