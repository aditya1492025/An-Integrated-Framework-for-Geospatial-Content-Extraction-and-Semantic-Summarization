import requests
from bs4 import BeautifulSoup
import streamlit as st

API_KEY = "your gcse api key"
CSE_ID = "your gcse id"

def search_google(query):
    search_url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={API_KEY}&cx={CSE_ID}"
    try:
        response = requests.get(search_url)
        response.raise_for_status()
        results = response.json()
        links = [item["link"] for item in results.get("items", []) if ".gov.in" in item["link"] or "pib.gov.in" in item["link"]]
        return links
    except Exception as e:
        print(f"Google Custom Search failed for {query}: {e}")
        return []

def scrape_text_from_url(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, verify=False)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = soup.find_all("p")
            return "\n".join([p.get_text() for p in paragraphs]).strip()
        return ""
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")
        return ""

def scrape_web_data(project_title):
    project_title_lower = project_title.lower()
    search_results = search_google(project_title)
    searched_urls = search_results.copy()  # All URLs searched
    contributing_urls = []  # URLs that contributed text
    
    if search_results:
        all_scraped_data = []
        for url in search_results[:5]: #for if u want to increase searches then change 5 to 10 or other
            page_text = scrape_text_from_url(url)
            if page_text:  # Only add if text is extracted
                contributing_urls.append(url)
                if project_title_lower in page_text.lower():
                    #st.info(f"Searched URLs: {', '.join(searched_urls)}\nContributing URL: {url}")
                    return page_text, searched_urls, [url]
                all_scraped_data.append(page_text)
        if all_scraped_data:
            combined_text = "\n\n".join(all_scraped_data)
            st.info(f"Searched URLs: {', '.join(searched_urls)}\nContributing URLs: {', '.join(contributing_urls)}")
            return combined_text, searched_urls, contributing_urls
    
    # Fallback URLs
    urls = ["https://pib.gov.in/indexd.aspx?reg=3&lang=1", "http://www.cspm.gov.in/"]
    searched_urls.extend(urls)  # Add to searched URLs
    fallback_data = []
    fallback_contributing_urls = []
    for url in urls:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text().lower()
            if text:
                fallback_contributing_urls.append(url)
                if project_title_lower in text:
                    st.info(f"Searched URLs: {', '.join(searched_urls)}\nContributing URL: {url}")
                    return text, searched_urls, [url]
                fallback_data.append(text)
        except Exception as e:
            print(f"Failed to scrape {url} for {project_title}: {e}")
    
    default_message = f"Information about {project_title} is limited. This project is under development with ongoing updates."
    if fallback_data:
        combined_fallback = "\n\n".join(fallback_data)
        st.info(f"Searched URLs: {', '.join(searched_urls)}\nContributing URLs: {', '.join(fallback_contributing_urls)}")
        return f"{default_message}\n\nAdditional context:\n{combined_fallback}", searched_urls, fallback_contributing_urls
    st.info(f"Searched URLs: {', '.join(searched_urls)}\nNo contributing URLs found; using default message.")
    return default_message, searched_urls, []
