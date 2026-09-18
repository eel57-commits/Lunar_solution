import requests
import json
import cv2
import numpy as np

# create dummy images
src = np.zeros((100, 100, 3), dtype=np.uint8)
ref = np.zeros((100, 100, 3), dtype=np.uint8)
cv2.imwrite('dummy_src.png', src)
cv2.imwrite('dummy_ref.png', ref)

files = {
    'source': open('dummy_src.png', 'rb'),
    'reference': open('dummy_ref.png', 'rb')
}

data = {
    'method': 'sift'
}

print("Testing API...")
response = requests.post('http://localhost:5000/register', files=files, data=data)
print(f"Status Code: {response.status_code}")
if response.status_code == 200:
    print("Success! Response:")
    print(response.json())
else:
    print("Error:")
    print(response.text)
