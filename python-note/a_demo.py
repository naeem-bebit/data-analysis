import sys

i = 0
while i < 5:
    i += 1
    if i == 3:
        break
    print('first', i, end=' ')

i = 0
while i < 5:
    i += 1
    if i == 3:
        continue
    # print('second', i)
    print()  # print new line

i = 0
while i < 5:
    print(i)
    i += 1
    if i == 3:
        continue
    print('third', i)
else:
    print('done')

# i = 0
for i in range(5):
    if i == 3:
        print(i * 3)
        continue
    print(i)
else:
    print('Done')

for i in range(3, 8, 2):
    print('fourth', i)

times2 = lambda a: a * 2
print('lam', times2(2))

# x = int(sys.argv[1]) # argv for the terminal
# y = int(sys.argv[2])
# print(x+y)

for i in range(4):
    for i in range(4):
        print('# ', end='')
    # print('\n')
    print()  # different with print new line

for i in range(4):
    for ix in range(1, i + 2):
        print(ix, end='')
    print()

for i in range(4):
    for ix in range(4 - i):
        print(ix, end='')
    print()


def add(x, y):
    c = x + y
    return c

result = add(2,3)
print(result)
print(add(1, 2))
