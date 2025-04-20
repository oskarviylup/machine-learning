import re
import time
from multiprocessing import Pool
from wsgiref import headers

import requests
import pandas as pd
from bs4 import BeautifulSoup
import cloudscraper

header = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
    'Referer': 'https://www.google.com'
}
header1 = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15',
    'Referer': 'https://ya.ru'
}
free_proxy_website_url = 'https://www.sslproxies.org'

s = cloudscraper.create_scraper(delay=10, browser={'custom': 'ScraperBot/1.0'})
params = {
    'forceLocation': False,
    'locationId': 653040,
    'lastStamp': 1683748131,
    'limit': 30,
    'offset': 89,
    'categoryId': 4
}

proxy_list = ['5.35.82.129:3128', '62.33.53.248:3128']
max_attempts = 2
processed_links = 0


def find_all_car_info(soup):
    pattern1 = re.compile(r'\d+')
    car_engine = soup.find('span', class_='css-1jygg09 e162wx9x0').text.split(",") if soup.find('span',
                                                                                                class_='css-1jygg09 e162wx9x0') else '--,--'
    try:
        engine = car_engine[1][:-2]
    except IndexError:
        engine = '---'
    car_probeg = soup.find('span', class_='css-1osyw3j ei6iaw00').text.split(",") if soup.find('span',
                                                                                               class_='css-1osyw3j ei6iaw00') else '--,--'
    car_hp = soup.find('span', class_='css-9g0qum e162wx9x0') if soup.find('span',
                                                                           class_='css-9g0qum e162wx9x0') else '---'
    try:
        hp = car_hp.text[:-12]
    except AttributeError:
        hp = '---'
    car_price = soup.find('div', class_='wb9m8q0') if soup.find('div', class_='wb9m8q0') else 'No price'
    try:
        price = car_price.text[:-2].replace('\xa0', ' ')
    except AttributeError:
        price = 'No price'

    car_name = soup.find('a', {'data-ftid': 'component_brand-model_title'}).get('title').split()[0] if soup.find('a', {
        'data-ftid': 'component_brand-model_title'}) else '---'
    car_year = soup.find('span', class_='css-1kb7l9z e162wx9x0').text if soup.find('span',
                                                                                   class_='css-1kb7l9z e162wx9x0') else '---'
    try:
        year = pattern1.findall(car_year)[0]
    except IndexError:
        year = '0'

    if soup.find('a', {'data-ga-stats-name': 'complectation_link'}):
        comlect = soup.find('a', {'data-ga-stats-name': 'complectation_link'})
        url1 = comlect.get('href')

        # Загружаем страницу
        response1 = requests.get(url1)
        # Создаем объект BeautifulSoup
        soup1 = BeautifulSoup(response1.content, 'html.parser')

        # Парсинг расхода
        try:
            volume_label = soup1.find('div', text='Расход')  # Находим элемент с текстом 'Расход'
            if volume_label:
                volume_value = volume_label.find_next('div').text  # Следующий элемент с текстом расхода
                c_volume = float(volume_value.replace('л', '').replace(',', '.').strip())
            else:
                c_volume = 0
        except (AttributeError, ValueError):
            c_volume = 0

        # Парсинг количества мест
        try:
            places_label = soup1.find('div', text='Кол-во мест')  # Находим элемент с текстом 'Кол-во мест'
            if places_label:
                places_quont = places_label.find_next('div').text[:-5]  # Следующий элемент с количеством мест
            else:
                places_quont = 0
        except TypeError:
            places_quont = 0

        # Парсинг трансмиссии
        try:
            trans_label = soup1.find('div', text='Трансмиссия')  # Находим элемент с текстом 'Трансмиссия'
            if trans_label:
                trans_value = trans_label.find_next('div').text  # Следующий элемент с текстом трансмиссии
            else:
                trans_value = '-'
        except AttributeError:
            trans_value = '-'
    else:
        c_volume = 0
        places_quont = '- мест'
        trans_value = '-----'

    return ({'Пробег': car_probeg[0][:-3].replace('\xa0', ' '),
             'Марка': car_name,
             'Объем двигателя': engine,
             'Год создания': year,
             'Тип двигателя': car_engine[0],
             'Цена': price,
             'Лошадиные силы': hp,
             'Расход': c_volume,
             'Кол-во мест': places_quont,
             'Трансмиссия': trans_value
             })


def parse_page(url):
    time.sleep(1)
    current_proxy_index = 0
    while current_proxy_index < max_attempts:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            return find_all_car_info(soup)
        else:
            proxy = proxy_list[current_proxy_index]
            proxies = {"http": f"http://{proxy}", "https": f"http://{proxy}"}
            try:
                response = requests.get(url, headers=header1, proxies=proxies)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    return find_all_car_info(soup)
                else:
                    current_proxy_index += 1
            except requests.RequestException:
                current_proxy_index += 1
    return ({'Пробег': 'err',
             'Марка': 'err',
             'Объем двигателя': 'err',
             'Год создания': 'err',
             'Тип двигателя': 'err',
             'Цена': 'err',
             'Лошадиные силы': 'err',
             'Расход': 'err',
             'Кол-во мест': 'err',
             'Трансмиссия': 'err'
             })


def parse_all_pages(urls):
    print("Parsing process started...")
    with Pool(processes=6) as pool:  # Multiprocessing
        results = pool.map(parse_page, urls)
    return results


if __name__ == '__main__':
    data = []

    url = 'https://spb.drom.ru/auto/used/all/'

    response1 = requests.get(url, headers=header)
    soup1 = BeautifulSoup(response1.text, 'html.parser')
    car_links1 = soup1.find_all('a', class_='g6gv8w4 g6gv8w8 _1ioeqy90')

    next_page_link = 'https://spb.drom.ru/auto/used/all/page2/'

    for i in range(2, 150):
        time.sleep(1)
        next_page_link_new = next_page_link.replace('2', str(i))
        response2 = requests.get(next_page_link_new, headers=header1)
        soup2 = BeautifulSoup(response2.text, 'html.parser')
        car_links1 += soup2.find_all('a', class_='g6gv8w4 g6gv8w8 _1ioeqy90')

    car_hrefs = [link.get('href') for link in car_links1]
    print(f"All links found. Quont: {len(car_hrefs)}")

    results = parse_all_pages(car_hrefs)

    ds1 = pd.DataFrame(results)
    ds1.to_csv('data_set_1.csv', index=False, encoding='utf-8')
    print(ds1)
