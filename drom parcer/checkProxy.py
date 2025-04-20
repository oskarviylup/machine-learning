import requests


def check_proxy(proxy):
    test_url = "https://httpbin.org/ip"
    proxies = {
        "http": f"http://{proxy}",
        "https": f"http://{proxy}"
    }

    try:
        response = requests.get(test_url, proxies=proxies, timeout=5)
        if response.status_code == 200:
            print(f"Прокси работает: {proxy} | Ответ: {response.json()}")
            return True
        else:
            print(f"Прокси не отвечает: {proxy}")
            return False
    except requests.RequestException as e:
        print(f"Ошибка подключения: {proxy} | Ошибка: {e}")
        return False

