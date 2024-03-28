import requests
import json
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import time

def sanitize_name(name):
    invalid_chars = ':*?"<>|\\/'
    for char in invalid_chars:
        name = name.replace(char, '')
    return name

def download_image(card):
    if 'image_uris' in card and 'multiverse_ids' in card and card['multiverse_ids']:
        card_name = sanitize_name(card['name'])
        multiverse_id = card['multiverse_ids'][0]
        folder_name = f"{card_name}-{multiverse_id}"
        folder_path = os.path.join("images", folder_name)
        os.makedirs(folder_path, exist_ok=True)

        image_url = card['image_uris']['large']
        image_path = os.path.join(folder_path, f"{card_name}-{multiverse_id}-large.jpg")
        if not os.path.exists(image_path):
            response = requests.get(image_url, stream=True)
            if response.status_code == 200:
                with open(image_path, 'wb') as f:
                    for chunk in response.iter_content(8192):
                        f.write(chunk)
            time.sleep(0.05)  # Sleep for 50 milliseconds

def fetch_and_download():
    url = "https://api.scryfall.com/cards/search?order=set&q=lang:en&unique=prints"
    cards = []
    while url:
        response = requests.get(url)
        data = response.json()
        cards.extend(data['data'])
        url = data.get('next_page')

    with ThreadPoolExecutor(max_workers=16) as executor:
        list(tqdm(executor.map(download_image, cards), total=len(cards)))

if __name__ == "__main__":
    fetch_and_download()
