import requests
from bs4 import BeautifulSoup

def get_mandi_bhav():
    url = "https://agmarknet.gov.in/"  # ya jis site se data chahiye
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36"
    }

    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        return {"error": f"Failed to fetch. Status code: {response.status_code}"}

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table")  # adjust selector as needed

    if not table:
        return {"error": "Table not found"}

    mandi_data = []
    for row in table.find_all("tr")[1:]:
        cols = row.find_all("td")
        if len(cols) >= 3:
            mandi_data.append({
                "commodity": cols[0].text.strip(),
                "location": cols[1].text.strip(),
                "price": cols[2].text.strip()
            })

    return mandi_data

# Run part
if __name__ == "__main__":
    data = get_mandi_bhav()

    if "error" in data:
        print("❌", data["error"])
    else:
        for item in data[:10]:  # top 10 entries
            print(f"{item['commodity']} - {item['location']} - ₹{item['price']}")

