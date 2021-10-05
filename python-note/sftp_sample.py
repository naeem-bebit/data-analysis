import pysftp

with pysftp.Connection(host="www.destination.com", username="root",
password="password",log="./temp/pysftp.log") as sftp:

  sftp.cwd('/root/public')  # The full path
  sftp.put('C:\Users\XXX\Dropbox\test.txt')
