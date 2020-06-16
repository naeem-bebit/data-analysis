from unittest.mock import Mock
import json
mock = Mock()
data = json.dumps({'a':1})

# print(data)
json = mock
json("a")
json.dumps({"a":1})


# print(json)
# print(dir(json))

print(json.assert_called_once())
print(json.dumps.assert_called_once())
# List Comprehension
nums = [1, 2, 3, 4]

a = []
for i in nums:
    a.append(i)
# print(a)

b = [i for i in nums]
# print(b)

c = [i*i for i in nums]
# print(c)

# map & lambda can be changed to list comprehension
# filter & lambda can be changed to list comprehension as well

d = [i for i in nums if i % 2 == 0]
# print(d)

letter = ['a', 'b', 'c', 'd']
# print(letter)
# print(type(letter))
abc = 'abc'
# print(type(abc))
asd = [i for i in abc]
# print(asd)

mix = [(i,d) for i in 'abc' for d in range(4)]
# print(mix)