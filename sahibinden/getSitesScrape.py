from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import time


chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

driver_path = "/mnt/c/Users/filiz/osuScrape/driver/chromedriver"

driver = webdriver.Chrome(executable_path=driver_path, chrome_options=chrome_options)

offset = 0
href= "https://www.sahibinden.com/volvo-s60"+"?pagingOffset="+str(offset)

driver.get(href)
driver.maximize_window()

total_page_web_element = driver.find_element_by_xpath('//*[@id="searchResultsSearchForm"]/div/div[4]/div[3]/p')
total_page_text = total_page_web_element.text
print(total_page_text)
total_page_size2scan = int(total_page_text.split()[1])
total_page_size2scan = int(total_page_size2scan / 10) * 10

href_set = set()

# for i in range(total_page_size2scan):
# no c like for for python
i = 0
while i < total_page_size2scan:
    try:
        offset = i * 20
        
        href= "https://www.sahibinden.com/volvo-s60"+"?pagingOffset="+str(offset)
        print("INFO : getting ", href)
        driver.get(href)
        driver.maximize_window()

        cars_table = driver.find_element_by_xpath('//*[@id="searchResultsTable"]')
        web_elements = cars_table.find_elements_by_xpath('.//a')

        element2scan = 0
        for web_element in web_elements:
            href=web_element.get_attribute('href')
            if href!=None and href.find("ilan") != -1:
                href_set.add(href)
                element2scan+=1
            # cars_table.find_elements_by_xpath('.//a')
        i+=1
    except:
        print("WARNING : ", str(offset) , " ban oluştu. Siteden izin alınana kadar bekleniyor.")
        
        href_set = set(list(href_set)[:-element2scan])

        try : 
            time.sleep(300)
        except:
            print("interrupt verild. Devam Ediliyor")
        i-=1
print(i)
driver.close()
driver.quit()
with open('sites.txt', 'w') as f:
    for item in list(href_set):
        f.write("%s\n" % item)
