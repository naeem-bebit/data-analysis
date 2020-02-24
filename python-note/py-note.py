"""Reference for Python code examples."""

# Dictionary True
n = int(input().strip())
check = {True: "Not Weird", False: "Weird"}
print(check[n % 2 == 0 and (n in range(2, 6) or n > 20)])

# integer list of integers combine into a integer
for i in range(1, (n + 1)):
    print(i, end="")
