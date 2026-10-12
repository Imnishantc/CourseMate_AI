import requests
from bs4 import BeautifulSoup

url = "https://www.samsung.com/in/"
response = requests.get(url, timeout=15)
response.raise_for_status()

soup = BeautifulSoup(response.text, "html.parser")
print(soup.get_text(" ", strip=True))
