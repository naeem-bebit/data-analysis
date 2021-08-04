from flask import Flask, request, jsonify

app = Flask(__name__)

countries = [
    {"id": 1, "name": "Thailand", "capital": "Bangkok", "area": 513120},
    {"id": 2, "name": "Australia", "capital": "Canberra", "area": 7617930},
    {"id": 3, "name": "Egypt", "capital": "Cairo", "area": 1010408},
]

def _find_next_id():
    return max(country["id"] for country in countries) + 1

@app.get("/countries")
def get_countries():
    return jsonify(countries)

@app.post("/countries")
def add_country():
    if request.is_json:
        country = request.get_json()
        country["id"] = _find_next_id()
        countries.append(country)
        return country, 201
    return {"error": "Request must be JSON"}, 415


curl --location --request POST 'https://api-lite.qr.live.goquo.io/api/v1/package/search?startSearch=true' \
--header 'x-access-site: 42fa0786-7f02-4a84-b675-3577c153b07e' \
--header 'x-access-token: Qatar Airways Holidays' \
--header 'Cookie: UserId=7b870581-b7f4-4434-be83-b613db306221; UserId=7b870581-b7f4-4434-be83-b613db306221' \
--header 'x-token: Q3it85M5lUWCApmSlpss3c0DrUZ4XVZK'


curl --location --request POST 'https://api-lite.mhh.live.goquo.io/api/v1/package/site-info' \
--header 'x-access-site: d4d864d5-43bf-4bd2-b846-19cb034f0087' \
--header 'x-access-token: MHholidays' \
--header 'x-token: Q3it85M5lUWCApmSlpss3c0DrUZ4XVZK' \
--header 'Cookie: UserId=56fdb175-d54d-4985-9eb1-3515482e46fa; UserId=56fdb175-d54d-4985-9eb1-3515482e46fa'

