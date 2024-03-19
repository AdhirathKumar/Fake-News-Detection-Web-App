from urllib.parse import urlparse


def check_iftrusted(url):
    trusted_sources = [
        "ndtv",
        "indiatoday",
        "thehindu",
        "news18",
        "deccanchronicle",
        "indiatimes",
        "hindustantimes",
        "livemint",
        "deccanherald",
        "thenewsminute",
        "nytimes",
        "cnn",
        "theguardian",
        "foxnews",
        "bbc",
        "dailymail",
        "washingtonpost",
        "wsj",
        "usatoday",
        "huffpost",
        "ndtv",
        "indiatoday",
        "indianexpress",
        "thehindu",
        "news18",
        "firstpost",
        "business-standard",
        "dna",
        "deccanchronicle",
        "storifynews",
        "hubnews",
        "newskibaat",
        "vindhyafirst",
        "blitzindia",
        "dailytimesindia",
        "sangritoday",
        "biovoicenews",
        "theprobe",
        "news89",
        "thetatva",
        "dailyexcelsior",
        "india.com",
        "oneindia",
        "mint",
        "indiatv",
        "patrika",
        "hindustantimes",
        "thebetterindia",
        "thewire",
        "deccanherald",
        "timesofindia",
    ]

    parsed_url = urlparse(url)
    domain_name_parts = parsed_url.netloc.split(".")

    if domain_name_parts[0] == "www":
        domain = domain_name_parts[1]
    else:
        domain = domain_name_parts[0]

    if domain in trusted_sources:
        return domain, True
    else:
        return domain, False
