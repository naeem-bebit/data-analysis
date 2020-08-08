i = 0
while i < 5:
    i += 1
    if i == 3:
        break
    print('first', i)

i = 0
while i < 5:
    i += 1
    if i == 3:
        continue
    print('second', i)

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
        print(i*3)
        continue
    print(i)
else:
    print('Done')

for i in range(3, 8, 2):
    print('fourth', i)

times2 = lambda a: a*2
print('lam', times2(2))

