import requests

def get_weather(city_name, api_key):
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    
    params = {
        'q': city_name,
        'appid': api_key,
        'units': 'metric'  # Celsius
    }

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.json()

        weather = {
            'City': data['name'],
            'Temperature (°C)': data['main']['temp'],
            'Humidity (%)': data['main']['humidity'],
            'Pressure (hPa)': data['main']['pressure'],
            'Weather Condition': data['weather'][0]['description'].title(),
            'Wind Speed (m/s)': data['wind']['speed']
        }

        return weather
    else:
        try:
            error_msg = response.json().get("message", "Unknown error")
        except:
            error_msg = response.text

        return {"Error": f"{response.status_code} - {error_msg}"}


# ------------------ Run Section ------------------

if __name__ == "__main__":
    API_KEY = "c91696aeafa15829ef4704736015b499"  # 👉 यहाँ अपनी सही API key डालना

    city = input("🌍 Enter your city/village name: ").strip()

    weather_data = get_weather(city, API_KEY)

    if "Error" not in weather_data:
        print("\n✅ Live Weather Report:")
        for key, value in weather_data.items():
            print(f"{key}: {value}")
    else:
        print("❌", weather_data["Error"])
