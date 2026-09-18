import requests

files = {
    'source': open('moon-base-equatorial (0).png', 'rb'),
    'reference': open('moon-base-equatorial (1).png', 'rb')
}

data = {
    'method': 'vit'
}

response = requests.post('http://localhost:5000/register', files=files, data=data)
print(response.status_code)
print(response.text)
