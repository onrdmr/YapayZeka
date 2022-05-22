from calendar import c
from this import d
import csv
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys


from spoofer import WebDriver

import re
import sys
import time



driver_path = "/mnt/c/Users/filiz/osuScrape/driver/chromedriver"


driver= WebDriver(driver_path)
driverinstance = driver.driver_instance

# stealth(driver,
#       languages=["en-US", "en"],
#       vendor="Google Inc.",
#       platform="Win32",
#       webgl_vendor="Intel Inc.",
#       renderer="Intel Iris OpenGL Engine",
#       fix_hairline=True,
#   )

# sys.argv[1]
sites_list = open(sys.argv[1]).readlines()

# tüm verisetin olacağı sözlük 
# her bir meta veri için liste olacak ve bu listeyi sonrakindekiler işeretlenerek yazılacak.
# örnein yeni bir işaret geldi diğer verilerin 1 eksii kadar 0 sonra asıl veri gelecektir.
dataset_dict = {}

#static veri seti meta datası websitesinden gelen 
dataset_dict['Ilan No'] = [] # ilan no kesin olacaktır.
dataset_dict['Hasar Kaydı'] = []
dataset_dict['Kimden'] = []
dataset_dict['Takas'] = []
dataset_dict['Renk'] = []
dataset_dict['KM'] = []
dataset_dict['Plaka'] = []
dataset_dict['Durumu'] = []
dataset_dict['Garanti'] = []
dataset_dict['Ilan Tarihi'] = []
dataset_dict['Vites'] = []
dataset_dict['Çekiş'] = []
dataset_dict['Fiyatı'] = []

# for idx in range(len(sites_list)):
# no c like for in python
# last_broken_place = open('idx.idx','r+')
idx = 0#int(last_broken_place.readline())
while idx < len(sites_list):
    site = sites_list[idx]
    print("INFO : Scraping ", site)
    try :
        time.sleep(0.05)
        driverinstance.get(site.strip())


        ad_detail = driverinstance.find_element_by_xpath('//*[@id="classifiedDescription"]')
        ad_detail_text = ad_detail.text

        # ilan no , ilan tarihi, marka, seri, model, yıl, Yakıt, vites, km, kasa tipi, motor gücvü, motor hacmi, Çekiş, renk, Garanti, Plaka, Kimden, takas, durumu
        base_list = driverinstance.find_element_by_xpath('//*[@id="classifiedDetail"]/div/div[2]/div[2]/ul')
        base_list_text = base_list.text
        base=base_list_text.split("\n")
        adv_no = base[1].strip()
        adv_date = base[3].strip()
        adv_brand = base[5].strip()
        adv_model = base[9].strip()
        adv_model_year = base[11].strip()
        adv_fuel = base[13].strip()
        adv_gear = base[15].strip()
        adv_km = base[17].strip()
        adv_case_type = base[19].strip()
        adv_engine_power = base[21].strip()
        adv_engine_volume = base[23].strip()
        adv_engine_bit = base[25].strip() # araba çekişi
        adv_color = base[27].strip()
        adv_quarantee = base[29].strip()
        adv_plate = base[31].strip()
        adv_from_person = base[33].strip()
        adv_exchange = base[37].strip()
        adv_state = base[39].strip()

        # car price
        car_price = driverinstance.find_element_by_xpath('//*[@id="classifiedDetail"]/div/div[2]/div[2]/h3')
        car_price = car_price.text.split('\n')[0].strip()

        try :
            technical_details_style_button = driverinstance.find_element_by_xpath('//*[@id="teknik-detaylar"]')
        except: 
            print("INFO :bu site teknik özellik içermemektedir.", site)
            idx+=1
            continue
        technical_details_hardware_button = driverinstance.find_element_by_xpath('//*[@id="technical-details"]/div/div[4]/ul/li[1]')


        # technical section is opened now we can read its content
        technical_details_style_button.send_keys(Keys.RETURN)
        # and the click hardware
        driverinstance.execute_script("arguments[0].click();",technical_details_hardware_button)

        technical_details_web_element = driverinstance.find_element_by_xpath('//*[@id="technical-details"]')
        time.sleep(0.01)
        technical_details_list = technical_details_web_element.find_elements_by_xpath('.//tr')
        



        hasTramerRegister = re.search(r'( TL)|(TL )', ad_detail_text, flags=re.IGNORECASE)
        if(hasTramerRegister == None):
            tramer_info = re.findall(r'((tramer|hasar).(kayd[ıiİ])?.(\w+))',ad_detail_text, flags=re.IGNORECASE)
            if tramer_info != []:
                for string in tramer_info[0]:
                    if(bool(re.search('var',string, re.IGNORECASE)) or bool(re.search('mevcut',string, re.IGNORECASE))):
                        hasTramerRegister = True
                if(hasTramerRegister == None):
                    hasTramerRegister = False
            else:
                # ek işlev düşünülecek
                # tramer_info = re.findall(r'(YOKTUR)',ad_detail_text, flags=re.IGNORECASE)
                hasTramerRegister = False
        else:
            hasTramerRegister = True

        dataset_dict['Ilan No'].append(adv_no) # ilan no kesin olacaktır.
        dataset_dict['Hasar Kaydı'].append(str(int(hasTramerRegister)))
        dataset_dict['Kimden'].append(adv_from_person)
        dataset_dict['Takas'].append(adv_exchange)
        dataset_dict['Renk'].append(adv_color)
        dataset_dict['KM'].append(adv_km)
        dataset_dict['Plaka'].append(adv_plate)
        dataset_dict['Durumu'].append(adv_state)
        dataset_dict['Garanti'].append(adv_quarantee)
        dataset_dict['Ilan Tarihi'].append(adv_date)
        dataset_dict['Vites'].append(adv_gear)
        dataset_dict['Çekiş'].append(adv_engine_bit)
        dataset_dict['Fiyatı'].append(car_price)

        car_idx = len(dataset_dict['Ilan No'])

        for web_element in technical_details_list:
            try :
                column_name = web_element.find_element_by_xpath('./td[1]').text
                column_name = column_name.split('(')[0] # alt bilgiler sonradan ayrlacak
                value_name = web_element.find_element_by_xpath('./td[2]').text
                if bool(dataset_dict.get(column_name)) == False:
                    dataset_dict[column_name] = []
                elif len(dataset_dict[column_name]) == car_idx:
                    continue
                    
                dataset_dict[column_name].append(value_name)
            except :
                print("WARNING : Misunderstading Error occured.")
                break

        technical_details_hardware_table = driverinstance.find_element_by_xpath('//*[@id="technical-details"]/div/div[4]/ul/li[2]')
        technical_details_hardware_list = technical_details_hardware_table.find_elements_by_xpath('.//tr')

        for web_element in technical_details_hardware_list:
            try :
                column_name = web_element.find_element_by_xpath('./td[1]').text
                value_name = "1"
                

                if bool(dataset_dict.get(column_name)) == False:
                    dataset_dict[column_name] = (car_idx-1)*['0']
                    dataset_dict[column_name].append(value_name)
                elif len(dataset_dict.get(column_name)) == car_idx - 1:
                    dataset_dict[column_name].append(value_name)
                elif len(dataset_dict.get(column_name)) == car_idx:
                    continue
                else :
                    for skip in range(car_idx - len(dataset_dict[column_name])-1):
                        dataset_dict[column_name].append('0')
                    dataset_dict[column_name].append(value_name)
                # en son kalanların hes 0 ile doldurulacak böylece tüm listelrin sayısı eşit olcak.
            except :
                print("INFO : Misunderstading Error occured.")
                continue
        
        # tüm elemanları eşitleme işlemi bu donanım için yapılır


        total_row_sz = car_idx
        for key in dataset_dict.keys():
            if len(dataset_dict[key]) != total_row_sz:
                for i in range(total_row_sz - len(dataset_dict[key])):
                    dataset_dict[key].append('0')

        idx+=1
    except:
        print("WARNING : Sahibinden Uyarı mesajı alındı.Bir süre Beklenecek. Eğer Başlamıyorsa Modeminizi Sıfırlayın.")
        input("modemi sıfırlayın ya da ilanı kontrol edin. \n")

        # with open('volvoDataset.csv', 'a') as csv_file:
        #     writer = csv.writer(csv_file)
        #     writer.writerow(list(dataset_dict.keys()))
        #     for i in range(len(dataset_dict['Ilan No'])):
        #         row_list = []
        #         for key in dataset_dict.keys():
        #             row_list.append(list(dataset_dict[key])[i])
        #         writer.writerow(row_list)
        # last_broken_place.seek(0)
        # last_broken_place.write(str(idx))
        # last_broken_place.seek(0)
        # last_broken_place.close()

        # Burada kes 
        # print("burada interrupt ile kes modemi sıfırla sonra yeniden dene.")

        # son sütunu temizel
        # for key in dataset_dict.keys():
        #     if len(dataset_dict[key]) == len(dataset_dict['Ilan No']):
        #         dataset_dict[key].pop()
        try : 
            driverinstance.close()
            driverinstance.quit() # for removal seesion id
            # time.sleep(300)
            print("INFO : Bekleme süresi bitti. Driver Yeniden Başlatılıyor")
            
            driver= WebDriver(driver_path)
            driverinstance = driver.driver_instance

            # stealth(driver,
            #     languages=["en-US", "en"],
            #     vendor="Google Inc.",
            #     platform="Win32",
            #     webgl_vendor="Intel Inc.",
            #     renderer="Intel Iris OpenGL Engine",
            #     fix_hairline=True,
            # )


        except:
            print("interrupt verildİ. Devam Ediliyor")

        # idx-=1


#remove unwanted key
dataset_dict.pop('',None)
# iki yerde de aynısı var 
# dataset_dict['Motor Tipi'] = dataset_dict['Motor Tipi'][::2]


with open('volvoDataset2.csv', 'a') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(list(dataset_dict.keys()))
    for i in range(len(dataset_dict['Ilan No'])):
        row_list = []
        for key in dataset_dict.keys():
            row_list.append(list(dataset_dict[key])[i])
        writer.writerow(row_list)

driverinstance.close()
driverinstance.quit()
# import csv
# lock = open("lock","r+")

# with open('volvoDataset.csv','a') as csv_file:
#     writer = csv.DictWriter(csv_file, fieldnames=list(dataset_dict.keys()))
#     if lock.readline() == '0':
#         lock.seek(0)
#         writer.writeheader()
#         lock.write("1")
#         lock.seek(0)
#     writer.writerow(dataset_dict)
# lock.close()
# driver.close()
