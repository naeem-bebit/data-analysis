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
