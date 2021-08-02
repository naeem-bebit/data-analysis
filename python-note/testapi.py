import requests
import json

From = "FRA"
Destination = "SEZ"
DepartureDate = "2021/7/23"
ReturnDate = "2021/7/30"
Currency = "EUR"

payload = "{'SessionId': 'b85e474b-3b69-4677-80fb-1334fc8b3901', 'ProductId':'334d00d6-8cdd-47ce-b9c9-86176f3b2eb5', 'From': '" + From + "', 'To': '" + Destination + "', 'DepartureDate': '" + DepartureDate + "', 'ReturnDate': '" + ReturnDate + "', 'Currency': '" + Currency + "', 'CultureCode': 'en-US', 'JourneyType': 1, 'CabinClass': 1, 'RoomCount': 1, 'PaxInfos': [{'AdultCount': 1}], 'RequestUrl': ''}"

headers = {
  'x-access-site': '42fa0786-7f02-4a84-b675-3577c153b07e',
  'x-access-token': 'Qatar Airways Holidays',
  'x-token': 'Q3it85M5lUWCApmSlpss3c0DrUZ4XVZK',
  'Cookie': 'UserId=7b870581-b7f4-4434-be83-b613db306221',
  'Accept-Encoding': 'gzip',
  'Content-Type': 'application/json',
  'Connection': 'keep-alive',
  'User-Agent': 'PostmanRuntime/7.26.8'
}
# requests("POST", "https://api-lite.qr.live.goquo.io/api/v1/package/search?startSearch=true", payload, headers)
response = requests.post("https://api-lite.qr.live.goquo.io/api/v1/package/search?startSearch=true", payload, headers=headers)
print(response.status_code)
json.loads(response.text)
# json.loads(response.content)
