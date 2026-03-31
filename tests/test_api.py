import requests

BASE_URL = "https://6k47viuqqq.ap-southeast-1.awsapprunner.com/"
ENDPOINT = f"{BASE_URL}/predict"

test_data = {
    "property_type": "Condominium/Apartment",
    "mukim": "Mukim Setapak",
    "scheme_name_area": "GENTING COURT",
    "tenure": "Leasehold",
    "land_parcel_area": 832.05,
    "unit_level": 12
}

print(f"Sending request to {ENDPOINT}...\n")

try:
    response = requests.post(ENDPOINT, json=test_data)

    if response.status_code == 200:
        result = response.json()
        price = result.get("estimated_price_rm", 0)
        print("Success.\n")
        print(f"Estimated Price: RM{round(price):,d}\n")
    else:
        print(f"Failed. Status code: {response.status_code}\n")
        print(f"Response: {response.text}\n")

except Exception as e:
    print(f"Connection Error: {e}\n")