lst = [1, 2, 3, 4, 5]


def odd_even_count(l1):
    odd = 0
    even = 0
    oddlist = []
    evenlist = []

    for i in l1:
        if i % 2 == 0:
            even += 1
            evenlist.append(i)
        else:
            odd += 1
            oddlist.append(i)
    return odd, even, evenlist, oddlist


odd1, even1, evenlist1, oddlist1 = odd_even_count(lst)
print(f'Even {even1}, Odd {odd1}')
print(f'List Even {evenlist1}, Odd list {oddlist1}')

print('[Fib]')

def fib(n):
    a = 0
    b = 1

    if n == 1:
        print(a)
    elif n <= 0:
        print('Not valid')
    else:

        print(a)
        print(b)

        for i in range(2, n):
            c = a+b
            a = b
            b = c
            if c >= 100:
                break
            print(c)

fib(-1)

print('[factorial]')

def fac(n):
    print(n)
    a = 1
    for i in range(1, n+1):
        a = a*i
    print(a)

fac(5)

## Recursive factorial

def fact(n):
    if n == 0:
        return 1
    return n * fact(n-1)

print('Recursive factorial: ', fact(5))


l1 = [1,2,3,4]

re_l = filter(lambda a: a%2==0, l1)
print(re_l)

rel2 = list(map(lambda a: a*2, l1))
print(rel2)

lam = lambda a: a*2
print(lam(2))

# Linear Search

l1 = [1,2,3,4,5,99,56]

def linear_search(l1,n):
    for i in l1:
        if i == n:
            print('Found', i)
            break
    else:
        print('Not found')
        # else:
        #     print('Not found')
# n = int(input("INPUT NUMber"))
n = 4
linear_search(l1,n)

## Bubble Sort

l2 = [52,6,13,1,4,9]

def sort_buble(l2):
    for i in range(len(l2)-1, 0, -1):
        for j in range(i):
            if l2[j]>l2[j+1]:
                l2[j],l2[j+1] = l2[j+1],l2[j]
                # temp = l2[j]
                # l2[j] = l2[j+1]
                # l2[j+1] = temp

sort_buble(l2)
print(l2)

print(l1[1])
# Binary Search

def b_search(l1, n):
    if n == 1:
        print(1)
    return n



