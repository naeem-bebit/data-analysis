import boto3
import xml.etree.ElementTree as ET
from datetime import datetime
import pandas as pd

s3 = boto3.resource('s3')
bucket = s3.Bucket('future-markets-prod-s3-datalakebucket')
objs = list(bucket.objects.filter(Prefix='Malaysia/Bookings/2021-06-21/QRH002936_20210621_150318.xml'))
# print(objs)

for obj in bucket.objects.filter(Prefix='Malaysia/Bookings/2021-06-21/QRH002936_20210621_150318.xml'):
    key = obj.key
    body = obj.get()['Body'].read()
    # print(body)
    parsed_xml = ET.fromstring(body)
    print(parsed_xml.findall("."))
    print(parsed_xml.findtext("booking/BookingDate"))
    print(parsed_xml.findtext("booking/LastModified"))
    print(parsed_xml.tag)
    print(parsed_xml.attrib)
    print(parsed_xml[0][1].text)
    for child in parsed_xml:
        for ch in child:
            print(ch.tag, ch.attrib)
        print(child.tag, child.attrib)

for i in pd.date_range(start="2021-02-01",end="2021-02-05"):
    print(i)

(datetime.now() - pd.to_datetime(date_test)) < timedelta(minutes=15)
if (datetime.now() - pd.to_datetime(date_test)) < timedelta(minutes=15):
    print('Yes')
else:
    print('No')

# Check two files for difference
with open('QRH002934_20210616_065718 2.xml', 'r') as file1:
    with open('QRH002934_20210613_214107 2.xml', 'r') as file2:
        difference = set(file1).difference(file2)
        print(difference)
difference.discard('\n')

with open('diff.txt', 'w') as file_out:
    for line in difference:
        file_out.write(line)
