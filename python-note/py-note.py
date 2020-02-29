"""Reference for Python code examples."""

# Dictionary True
n = int(input().strip())
check = {True: "Not Weird", False: "Weird"}
print(check[n % 2 == 0 and (n in range(2, 6) or n > 20)])

#  convert a list of integers into a single integer
for i in range(1, (n + 1)):
    print(i, end="")

int("".join(map(str, range(1, (int(input())+1)))))
