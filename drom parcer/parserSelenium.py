import re
from multiprocessing import Pool
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException

# Функция для парсинга одной страницы
def parse_page(url):
    pattern1 = re.compile(r'\d+')

    # Настройки для работы в headless режиме
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Режим без графического интерфейса

    # Инициализация драйвера
    driver = webdriver.Chrome(service=Service(), options=chrome_options)
    driver.get(url)

    try:
        car_engine = driver.find_element(By.CLASS_NAME, 'css-1jygg09.e162wx9x0').text.split(",") if driver.find_element(By.CLASS_NAME, 'css-1jygg09.e162wx9x0') else '--,--'
    except NoSuchElementException:
        car_engine = '--,--'

    try:
        car_probeg = driver.find_element(By.CLASS_NAME, 'css-1osyw3j.ei6iaw00').text.split(",") if driver.find_element(By.CLASS_NAME, 'css-1osyw3j.ei6iaw00') else '--,--'
    except NoSuchElementException:
        car_probeg = '--,--'

    try:
        car_hp = driver.find_element(By.CLASS_NAME, 'css-9g0qum.e162wx9x0') if driver.find_element(By.CLASS_NAME, 'css-9g0qum.e162wx9x0') else '---'
        hp = car_hp.text[:-12]
    except NoSuchElementException:
        hp = '---'

    try:
        car_price = driver.find_element(By.CLASS_NAME, 'wb9m8q0') if driver.find_element(By.CLASS_NAME, 'wb9m8q0') else 'No price'
        price = car_price.text[:-2].replace('\xa0', ' ')
    except NoSuchElementException:
        price = 'No price'

    try:
        car_name = driver.find_element(By.XPATH, "//a[@data-ftid='component_brand-model_title']").get_attribute('title').split()[0]
    except NoSuchElementException:
        car_name = '---'

    try:
        car_year = driver.find_element(By.CLASS_NAME, 'css-1kb7l9z.e162wx9x0').text
        year = pattern1.findall(car_year)[0]
    except (NoSuchElementException, IndexError):
        year = '0'

    # Проверяем наличие комплектации и парсим доп. параметры
    try:
        comlect = driver.find_element(By.XPATH, "//a[@data-ga-stats-name='complectation_link']")
        url1 = comlect.get_attribute('href')

        driver.get(url1)

        # Парсинг расхода
        try:
            volume_label = driver.find_element(By.XPATH, "//div[contains(text(), 'Расход')]")
            volume_value = volume_label.find_element(By.XPATH, "following-sibling::div").text
            c_volume = float(volume_value.replace('л', '').replace(',', '.').strip())
        except (NoSuchElementException, ValueError):
            c_volume = 0

        # Парсинг количества мест
        try:
            places_label = driver.find_element(By.XPATH, "//div[contains(text(), 'Кол-во мест')]")
            places_quont = places_label.find_element(By.XPATH, "following-sibling::div").text
        except NoSuchElementException:
            places_quont = 0

        # Парсинг трансмиссии
        try:
            trans_label = driver.find_element(By.XPATH, "//div[contains(text(), 'Трансмиссия')]")
            trans_value = trans_label.find_element(By.XPATH, "following-sibling::div").text
        except NoSuchElementException:
            trans_value = '-'
    except NoSuchElementException:
        c_volume = 0
        places_quont = '- мест'
        trans_value = '-----'

    driver.quit()  # Закрываем драйвер после выполнения парсинга страницы

    # Возвращаем результат как словарь
    return {
        'Пробег': car_probeg[0][:-3].replace('\xa0', ' '),
        'Марка': car_name,
        'Объем двигателя': car_engine[1][:-2] if len(car_engine) > 1 else '---',
        'Год создания': year,
        'Тип двигателя': car_engine[0],
        'Цена': price,
        'Лошадиные силы': hp,
        'Расход': c_volume,
        'Кол-во мест': places_quont[:-5] if places_quont != 0 else '---',
        'Трансмиссия': trans_value
    }


# Функция для парсинга всех страниц
def parse_all_pages(urls):
    with Pool(processes=6) as pool:  # Мультипроцессинг с 6 процессами
        results = pool.map(parse_page, urls)
    return results

if __name__ == '__main__':

    # Инициализация Selenium
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Запуск в headless режиме (без графического интерфейса)
    driver = webdriver.Chrome(service=Service(), options=chrome_options)

    # URL первой страницы
    url = 'https://spb.drom.ru/auto/used/all/'
    driver.get(url)

    # Сбор ссылок на машины
    car_links1 = driver.find_elements(By.XPATH, "//a[@class='g6gv8w4 g6gv8w8 _1ioeqy90']")
    print(car_links1)
    next_page_link = driver.find_element(By.XPATH, "//a[@class='_1j1e08n0 _1j1e08n5']")


    # Сбор ссылок с нескольких страниц
    while len(car_links1) <= 100:
        driver.get(next_page_link.get_attribute('href'))
        car_links1.append(driver.find_elements(By.XPATH, "//a[@class='g6gv8w4 g6gv8w8 _1ioeqy90']"))
        next_page_link = driver.find_element(By.XPATH, "//a[@class='_1j1e08n0 _1j1e08n5']")


    car_hrefs = [link.get_attribute('href') for link in car_links1]
    print(car_hrefs)

    # Запуск парсинга всех собранных ссылок
    results = parse_all_pages(car_hrefs)

    # Сохранение в CSV
    ds = pd.DataFrame(results)
    ds.to_csv('data_set_1.csv', index=False, encoding='utf-8')
    print(ds)
