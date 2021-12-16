## https://docs.python.org/3/library/tempfile.html

import tempfile
  
temp = tempfile.TemporaryFile()
print(temp)
print(temp.name)

temp = tempfile.NamedTemporaryFile()
print(temp)
print(temp.name)

temp = tempfile.NamedTemporaryFile(prefix='pre_', suffix='_suf')
print(temp.name)


with tempfile.TemporaryFile() as fp:
    fp.write(b'Hello world!')
    fp.seek(0)
    fp.read()
