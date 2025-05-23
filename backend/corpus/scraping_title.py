import requests
import time
import json
import os
from bs4 import BeautifulSoup
import threading

# Fungsi untuk mendapatkan ID dari URL
def get_id(url):
    return url.split('/')[-2]

# Fungsi untuk mendapatkan ringkasan dari artikel
def get_summary(text):
    target = ''
    for line in text.split('\n'):
        if 'window.kmklabs.channel =' in line:
            target = line
            break
    temp = target.split('window.kmklabs.article = ')[1]
    temp = temp.split(';')[0]
    data = json.loads(temp)
    return data['shortDescription']

# Fungsi untuk mengekstrak data dari HTML
def extract_data(text):
    soup = BeautifulSoup(text, 'html.parser')
    
    title = soup.find('title').getText().replace(' - News Liputan6.com', '') if soup.find('title') else 'No Title'
    
    date = soup.find('time', {'class': 'read-page--header--author__datetime updated'})
    date = date.getText() if date else 'No Date'
    
    article = []
    contents = soup.findAll('div', {'class': 'article-content-body__item-content'})
    for content in contents:
        article.append(content.getText())
    
    summary = get_summary(text) if text else 'No Summary'
    
    return title, date, article, summary

# Fungsi untuk menulis data ke file JSON
def write_file(id, url, title, date, content, summary, target_path):
    file_path = os.path.join(target_path, f"{id}.json")
    if os.path.exists(file_path):  # Jika file sudah ada, tidak perlu menulis ulang
        print(f"File {file_path} sudah ada. Lewatkan.")
        return
    
    # Menyimpan data dalam format yang sesuai untuk Hugging Face
    json_dict = {}
    json_dict['content'] = '\n'.join(content)  # Konten artikel
    json_dict['summary'] = summary  # Ringkasan artikel
    json_dict['id'] = id  # ID artikel
    json_dict['url'] = url  # URL artikel

    # Simpan sebagai file JSON
    with open(file_path, 'w') as json_file:
        json.dump(json_dict, json_file)

# Fungsi untuk memproses satu URL
def proceed_one(url, path):
    response = requests.get(url, allow_redirects=True)
    
    if response.status_code != 200:
        print(f"Failed to retrieve {url}. HTTP Status Code: {response.status_code}")
        return
    
    url = response.url 
    id = get_id(url)
    try:
        title, date, article, summary = extract_data(response.text)
        write_file(id, url, title, date, article, summary, path)
    except Exception as e:
        print(f"Error processing {url}: {e}")

# Fungsi untuk memproses URL dalam batch
def proceed(urls, path):
    for url in urls:
        try:
            print(f"Processing: {url}")
            proceed_one(url, path)
            time.sleep(2)  # Waktu tunggu antara permintaan
        except Exception as e:
            print(f"Failed to proceed {url}. Error: {e}")

# Fungsi untuk memulai thread dan memproses URL
def thread_func(urls, path, num_thread=1):
    os.makedirs(path, exist_ok=True)
    threads = []
    
    # Load URL terakhir yang sudah diproses jika ada
    last_processed = load_last_processed('last_processed.json')
    
    # Batasi URL yang sudah diproses
    urls = [url for url in urls if get_id(url) not in last_processed]
    
    # Bagi URL ke beberapa thread
    for i in range(num_thread):
        cur_idx = int(i * len(urls) / num_thread)
        cur_urls = urls[cur_idx:cur_idx + int(len(urls) / num_thread)]
        t = threading.Thread(target=proceed, args=(cur_urls, path,))
        threads.append(t)
        t.start()
    
    # Menunggu semua thread selesai sebelum melanjutkan
    for t in threads:
        t.join()
    
    # Update state setelah selesai
    update_last_processed('last_processed.json', urls)

# Fungsi untuk menyimpan state URL terakhir yang sudah diproses
def save_last_processed(filename, processed_urls):
    with open(filename, 'w') as f:
        json.dump({"processed_urls": processed_urls}, f)

# Fungsi untuk memuat state URL terakhir yang sudah diproses
def load_last_processed(filename):
    if not os.path.exists(filename):
        return set()  # Jika file tidak ada, kembalikan set kosong
    with open(filename, 'r') as f:
        data = json.load(f)
        return set(data.get("processed_urls", []))

# Fungsi untuk mengupdate state URL terakhir yang sudah diproses
def update_last_processed(filename, urls):
    processed_urls = [get_id(url) for url in urls]
    save_last_processed(filename, processed_urls)

# Konfigurasi dan menjalankan threading
THREAD = 10
urls = json.load(open('corpus/url.json'))

thread_func(urls['dev_urls'], 'data/raw/dev', THREAD)
thread_func(urls['test_urls'], 'data/raw/test', THREAD)
thread_func(urls['train_urls'], 'data/raw/train', THREAD)
