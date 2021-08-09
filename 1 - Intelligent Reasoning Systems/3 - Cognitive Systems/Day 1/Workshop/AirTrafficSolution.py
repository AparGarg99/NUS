# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#pip install beautifulsoup4

from bs4 import BeautifulSoup
import requests

def get(url):
    headers={}
    resp = requests.get(url, headers=headers)
    if resp.ok:
        return resp.text
    
#Extract the main content from each web url
def colcontent(url):
    data2 = get(url)
    soup = BeautifulSoup(data2,  "html.parser")
    maindiv = soup.find("div", {"class":"c-post-typography cell smedium-10 large-8 xlarge-7 contain-lead"})
    if maindiv:
        return (maindiv.text)

website = "https://www.airport-technology.com/company-a-z/"
data = get(website)

#Access all the relevant urls and store them in a list
soup = BeautifulSoup(data,"html.parser")
alltags = soup.findAll("a")

urls = []
names = []
for a_tag in alltags:
    href = a_tag.get("href")
    if href and href != "" and "contractors" in href:
        span_tag = str(a_tag.contents[0])
        if span_tag:
            clean = span_tag.strip(' ')
#            print(clean)
#            print(href)
            names.append(clean)
            urls.append(href)
            
       
totalCount = 20      
file = open("Air Traffic Control Solutions.txt","w",encoding='utf-8')

for item in urls[11:11+totalCount]:
    contents=colcontent(item)
    file.write(names[urls.index(item)])
    file.write(contents)
    print (names[urls.index(item)]+" downloaded.")

file.close()