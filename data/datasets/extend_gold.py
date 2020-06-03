import ast
import pandas as pd
import sys, os
import newspaper
from urllib.parse import urlparse

cwd = os.path.abspath(__file__+"/..")
html_folder = os.path.abspath(cwd+"/html_snopes/")
cookie_wall_origins_list = []
cookie_wall_origins_filename = os.path.abspath(cwd+"/cookie_wall_origins.txt")

agreement_keywords = ["agree and continue", "terms of service", "accept and continue", "This timeline is where you’ll spend", "we use cookies", "Reproduction is permitted with or without attribution"]

#page    claim   claim_label     tags    origin_list     date
gold_df = pd.read_csv("datasetVeritas3.csv", sep=",", header=0, names=["a_url","claim","verdict","a_tags","a_date","a_author","source_list","o_url", "value", "name"])
consolidated_gold = pd.DataFrame(columns=["a_url","claim","verdict","a_tags","a_date","a_author","source_list","o_url","o_domain","o_body","o_title","o_date","o_author","o_keywords","o_summary"])

cookie = 0
errors = 0
#o_domain,o_body,o_title,o_date,o_author,o_keywords,o_summary
for i,e in list(gold_df.iterrows()):
    print(i)
    #unfold the entry into the varibles
    a_url,claim,verdict,a_tags,a_date,a_author,source_list,o_url,value,name = e

    if len(a_author) < 2:
        print("author:",a_author)
        sys.exit(1)

    #cast string to list
    source_list = ast.literal_eval(e['source_list'])

    #finds which is the position of the o_url in the list (we will need that to retrieve the correct .html)
    o_idx = source_list.index(o_url)
    a = newspaper.Article(o_url)

    #finds the html file
    article_alias = a_url.rstrip("/").split("/")[-1]
    article_folder = html_folder+"/"+article_alias
    o_html_filename = article_folder+"/"+str(o_idx)+".html"

    # set html manually
    with open(o_html_filename, 'rb') as fh:
        a.html = fh.read()
        # need to set download_state to 2 for this to work
        a.download_state = 2
        a.parse()

    o_body = a.text
    #if text is inexistent, try downloading it
    if (len(o_body) < 100):
        try:
            a.download_state = 0
            a.download()
            a.parse()
        except:
            errors =+ 1
            continue

    o_body = a.text

    #check if the text is a 'cookie' or 'terms of service' message, if so, saves it to a txt
    cookie_flag = sum([1 for e in agreement_keywords if e in o_body.lower()])
    if cookie_flag > 0 and (o_url not in cookie_wall_origins_list):
        cookie += 1
        cookie_wall_origins_list.append(o_url)
        continue

    # gets the missing info: (o_url,o_domain,o_body,o_title,o_date,o_author,o_keywords,o_summary)
    # and adds the entry to the consolidated dataframe
    a.nlp()
    o_title = a.title
    o_date = a.publish_date
    if a_author is "":
        o_author = a.authors
    else:
        o_author = a_author
    o_tags = a.tags
    o_keywords = a.keywords
    o_summary = a.summary

    parsed_uri = urlparse(o_url)
    o_domain = '{uri.scheme}://{uri.netloc}/'.format(uri=parsed_uri)

    entry = [a_url,claim,verdict,a_tags,a_date,a_author,source_list,o_url,o_domain,o_body,o_title,o_date,o_author,o_keywords,o_summary]
    consolidated_gold.loc[len(consolidated_gold)] = entry

#saves the 'cookie wall origins" to a txt
for o_url in cookie_wall_origins_list:
    with open(cookie_wall_origins_filename, 'a+') as f:
        f.write(o_url+"\n")

print("cookie:", cookie)
print("errors:", errors)
print("size of df:", len(consolidated_gold))
consolidated_gold.to_csv("veritas4.1.csv", index=False, sep="\t")
input("Press ENTER")
