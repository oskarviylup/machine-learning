import requests
from bs4 import BeautifulSoup

from checkProxy import check_proxy


def proxy_parser(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    proxy_list = []
    for row in soup.find("table", {"id": "proxylister-table"}).find_all("tr"):
        cols = row.find_all("td")
        if len(cols) > 1 and cols[4].text == "elite proxy":  # Фильтрация по анонимности
            proxy = f"{cols[0].text}:{cols[1].text}"
            proxy_list.append(proxy)

    working_proxies = [proxy for proxy in proxy_list if check_proxy(proxy)]
    return working_proxies
proxy_list = [
    '91.202.197.1:3128',
    '185.105.89.174:8080',
    '46.47.197.210:3128',
    '82.146.37.145:80',
    '83.219.145.108:3128',
    '77.87.100.22:60606',
    '91.222.238.112:80',
    '94.159.17.26:1080',
    '185.221.152.148:3128',
    '176.119.20.81:8534',
    '188.244.38.134:7999',
    '217.15.149.192:8080',
    '46.39.21.101:60606',
    '77.222.58.239:14541',
    '188.130.240.136:8080',
    '94.142.141.145:1080',
    '46.254.220.30:8080',
    '62.183.96.194:8080',
    '217.197.121.35:3128',
    '176.119.19.26:60606',
    '5.35.82.129:3128',
    '147.45.166.62:1080',
    '5.228.191.67:8081',
    '213.251.252.89:1080',
    '94.230.127.180:1080',
    '62.33.53.248:3128',
    '95.66.138.21:8880',
    '95.163.20.1:4153',
    '62.182.204.81:88',
    '5.227.210.157:8424',
    '31.163.204.156:8080',
    '31.129.147.102:1080',
    '194.190.169.197:3701',
    '217.168.76.83:3128',
    '94.159.10.42:1080'
]

working_proxies = [proxy for proxy in proxy_list if check_proxy(proxy)]
print(working_proxies)


'''global current_proxy_index, processed_links
    while current_proxy_index < max_attempts:
        proxy = proxy_list[current_proxy_index]
        proxies = {"http": f"http://{proxy}", "https": f"http://{proxy}"}
        try:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                return find_all_car_info(soup)
            else:
                current_proxy_index += 1
        except requests.RequestException:
            current_proxy_index += 1
            '''