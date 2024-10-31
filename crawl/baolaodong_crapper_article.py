from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import csv
import re
import pandas as pd
import os

def process_html_content(tag_content):
    desired_tags = [
        "h1", "h2", "h3", "h4", "h5", "h6",
        "p",
        "br",
        "hr",
        "strong",
        "em",
        "b",
        "i",
        "small",
        "mark",
        "sub",
        "sup",
        "ul",
        "ol",
        "li",
        "span",
        "div"
    ]

    soup = BeautifulSoup(tag_content, "html.parser")

    for element in soup.find_all(True):
        if element.name not in desired_tags:
            element.decompose()

    text = soup.get_text(separator=" ")
    text = re.sub(r'[^\w\s,.ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯăạảấầẩẫậắằẳẵặẹẻẽềểếễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]', '', text)
    text = re.sub(
        r'[\U0001F600-\U0001F64F'
        r'\U0001F300-\U0001F5FF'
        r'\U0001F680-\U0001F6FF'
        r'\U0001F700-\U0001F77F'
        r'\U0001F780-\U0001F7FF'
        r'\U0001F800-\U0001F8FF'
        r'\U0001F900-\U0001F9FF'
        r'\U0001FA00-\U0001FA6F'
        r'\U00002700-\U000027BF'
        r'\U00002000-\U0000201F'
        r'\U00002500-\U0000257F'
        r'\U00002760-\U0000277F'
        r'\U0001F1E6-\U0001F1FF'
        r'\U0001F9C0-\U0001F9FF'
        r']+', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    for h_tag in desired_tags[:6]:
        for element in soup.find_all(h_tag):
            if not element.get_text().strip().endswith('.'):
                element.string = element.get_text().strip() + '.'

    return text

print('Đang cài đặt ChromeDriver...')

service = Service('/usr/bin/chromedriver')
options = webdriver.ChromeOptions()
options.add_argument("--headless")
driver = webdriver.Chrome(service=service, options=options)

tags = ["xa-hoi", "kinh-doanh", "the-gioi", "thoi-su", "the-thao", "van-hoa-giai-tri", "bat-dong-san", "cong-nghe", "viec-lam", "giao-duc"]
pattern_url = "https://laodong.vn/{tag}?page={page}"
pattern_prefix = "https://laodong.vn/{tag}/"
range_page = [2, 500]
file_results = 'baolaodong_articles.csv' # [tag, link, summary, content]
class_summary = 'chappeau' #  div tag
id_content = 'gallery-ctt' # div tag
total_links_collected = 0

if not os.path.exists(file_results):
    with open(file_results, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['tag', 'link', 'content', 'summary'])

for page in range(range_page[0], range_page[1] + 1):
    for tag in tags:
        url = pattern_url.format(tag=tag, page=page)
        print(f'Đang lấy link từ trang {url}...')
        try:
            driver.get(url)
            time.sleep(3)  # Đợi trang tải
            html_content = driver.page_source
            soup = BeautifulSoup(html_content, 'html.parser')
            links = soup.find_all('a', class_='link-title')
            filtered_links = [link['href'] for link in links if link['href'].startswith(pattern_prefix.format(tag=tag))]
            try :
                for link in filtered_links:
                    print('Đang tải', link)
                    driver.get(link)
                    time.sleep(3)
                    
                    html_content = driver.page_source
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    links = soup.find_all('a', class_='link-title')
                    filtered_links = [link['href'] for link in links if link['href'].startswith(pattern_prefix.format(tag=tag))]
                    
                    content_element = soup.find("div", id=id_content)
                    summary_element = soup.find("div", class_=class_summary)
                    
                    if content_element is None or summary_element is None:
                        print(f'Không tìm thấy nội dung hoặc tóm tắt cho link {link}')
                        continue
                    
                    content = process_html_content(str(content_element))
                    summary = process_html_content(str(summary_element))
                    
                    print('Đã tải xong', link)
                    
                    with open(file_results, mode='a', newline='', encoding='utf-8') as file:
                        writer = csv.writer(file)
                        writer.writerow([tag, link, content, summary])
                        total_links_collected += 1
            except Exception as e:
                print(f'Có lỗi xảy ra khi lấy trang {link}: {e}')
                continue
        except Exception as e:
            print(f'Có lỗi xảy ra khi lấy trang {url}: {e}')
            continue

driver.quit()
print("Quá trình hoàn tất, kết quả đã lưu vào", file_results)