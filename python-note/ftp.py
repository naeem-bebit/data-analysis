import ftplib

# https://dlptest.com/ftp-test/
FTP_HOST = "ftp.dlptest.com"
FTP_USER = "dlpuser"
FTP_PASS = "rNrKYTX9g7z3RgJRmxWuGHbeu"

ftp = ftplib.FTP(FTP_HOST, FTP_USER, FTP_PASS)
ftp.encoding = "utf-8"

ftp.dir()

# Upload a file
filename = "test.txt"
with open(filename, "rb") as file:
    ftp.storbinary("STOR test.txt", file)

# #Download a file
filename = 'test.txt'
with open(filename, "wb") as file:
    ftp.retrbinary("RETR test.txt", file.write)


ftp.quit()