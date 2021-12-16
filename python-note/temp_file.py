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

temp = tempfile.TemporaryFile()
temp.write(b'foo bar')
temp.seek(0)
print(temp.read())
temp.close()

temp_dir = tempfile.TemporaryDirectory()
print(temp_dir)

secure_temp = tempfile.mkstemp(prefix="pre_",suffix="_suf")
print(secure_temp)
  
tempfile.tempdir = "/temp"
print(tempfile.gettempdir())

with tempfile.TemporaryFile() as fp:
    fp.write(b'Hello world!')
    fp.seek(0)
    fp.read()
