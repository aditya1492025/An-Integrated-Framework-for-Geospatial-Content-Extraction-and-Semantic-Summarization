import re
import pdfplumber
import spacy
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

# Initialize geocoder
geolocator = Nominatim(user_agent="location_finder")

# Load spaCy models
def load_models():
    trained_nlp = spacy.load("Trained_spacy_model/model-best")
    untrained_nlp = spacy.load("en_core_web_trf")
    return trained_nlp, untrained_nlp

# Load predefined locations dataset
def load_predefined_locations():
    df = pd.read_csv("predefined_locations.csv")
    return (
        set(df["STATE/UT NAME"].dropna().str.lower().unique()),
        set(df["DISTRICT NAME"].dropna().str.lower().unique()),
        set(df["SUB-DISTRICT NAME"].dropna().str.lower().unique()),
        set(df["Area/Town Name"].dropna().str.lower().unique())
    )

# Rule-based location extraction
def rule_based_location_extraction(text, known_states, known_districts, known_subdistricts, known_towns):
    text_lower = text.lower().strip()
    invalid_terms = ["agreement", "committee", "response", "govt.", "issue", "price", "level", 
                     "thermal", "ltd", "secretary", "pipeline", "site", "township", "refinery", "crude", 
                     "tpp", "is", "o", "infrastructure", "project", "new", "line", "mp", "cr", "crore", 
                     "beneficiaries"]
    invalid_patterns = [r"^\d+$", r"\d+\s*(ha\.|mu|cr\.|rs\.|%|cr|crore|beneficiaries)", r"^\d+\.\d+$", 
                        r"^\d+\.\d+\.\d+", r"\d.*(scheme|ca|ha|km|has|cr|crore)", r".*\d+\s*\.\s*\d+"]
    if (len(text.split()) > 3 or len(text) < 2 or text_lower in invalid_terms or
        any(re.search(pattern, text_lower) for pattern in invalid_patterns)):
        return None
    if text_lower in known_states:
        return "STATE"
    elif text_lower in known_districts:
        return "DISTRICT"
    elif text_lower in known_subdistricts:
        return "SUBDISTRICT"
    elif text_lower in known_towns:
        return "TOWN"
    return None

# Extract entities using trained model
def extract_trained_entities(text, model, known_states, known_districts, known_subdistricts, known_towns):
    doc = model(text)
    results = []
    for ent in doc.ents:
        if ent.label_ in ("STATE", "DISTRICT", "SUBDISTRICT", "TOWN"):
            text = ent.text.strip()
            label = rule_based_location_extraction(text, known_states, known_districts, known_subdistricts, known_towns)
            if label and final_filter(text, label):
                results.append((text, label))
    return results

# Extract entities using untrained model
def extract_untrained_gpe_entities(text, model, known_states, known_districts, known_subdistricts, known_towns):
    doc = model(text)
    results = []
    for ent in doc.ents:
        if ent.label_ == "GPE":
            text = ent.text.strip()
            label = rule_based_location_extraction(text, known_states, known_districts, known_subdistricts, known_towns)
            if label and final_filter(text, label):
                results.append((text, label))
    return results

# Process text for locations
def process_text_for_locations(text, trained_nlp, untrained_nlp, known_states, known_districts, known_subdistricts, known_towns, limit=None):
    text_parts = re.split(r"[-/]", text)
    all_locations = set()
    for part in text_parts:
        part = part.strip()
        trained_results = extract_trained_entities(part, trained_nlp, known_states, known_districts, known_subdistricts, known_towns)
        all_locations.update(trained_results)
        untrained_results = extract_untrained_gpe_entities(part, untrained_nlp, known_states, known_districts, known_subdistricts, known_towns)
        all_locations.update(untrained_results)
    valid_locations = set()
    for loc, label in all_locations:
        if final_filter(loc, label):
            valid_locations.add((loc, label))
    sorted_locations = sorted(valid_locations, key=lambda x: (["STATE", "DISTRICT", "SUBDISTRICT", "TOWN"].index(x[1]), x[0]))
    if limit is not None:
        return sorted_locations[:limit]
    return sorted_locations

# Extract text from PDF
def extract_text(pdf_path):
    text_content = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_content += page_text + "\n"
    return text_content

# Extract title from PDF
def extract_title(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        first_page = pdf.pages[0]
        
        # Step 1: Try the first line of the first page
        text = first_page.extract_text()
        if text:
            first_line = text.split('\n')[0].strip()
            if first_line and len(first_line) > 2:  # Ensure it’s not too short
                return first_line
        
        # Step 2: Fallback to centered text with "Project" keyword
        words = first_page.extract_words()
        page_width = first_page.width
        title_lines = {}
        for word in words:
            y_pos = word["top"]
            x_center = (word["x0"] + word["x1"]) / 2
            text_content = word["text"]
            if page_width * 0.3 < x_center < page_width * 0.7:  # Centered text
                y_key = round(y_pos / 10) * 10
                if y_key not in title_lines:
                    title_lines[y_key] = []
                title_lines[y_key].append((x_center, text_content))
        
        candidate_lines = [(y_key, " ".join(word[1] for word in sorted(title_lines[y_key], key=lambda x: x[0]))) 
                           for y_key in sorted(title_lines.keys())]
        for _, line in candidate_lines:
            if any(keyword in line for keyword in ["Project", "PROJECT", "project"]):
                return line.strip()
        
        # Step 3: Fallback to the first centered line if no "Project" keyword
        title_text = candidate_lines[0][1] if candidate_lines else None
        return title_text.strip() if title_text else None

# Extract tables from PDF
def extract_tables(pdf_path):
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_tables = page.extract_tables()
            if page_tables:
                tables.extend(page_tables)
    return tables

# Extract locations from tables
def extract_location_from_tables(tables, known_states, known_districts, known_subdistricts, known_towns):
    if not tables:
        return None
    locations = []
    location_keywords = ["Location", "Project Location", "Area"]
    found_location_section = False
    for table in tables:
        for row in table:
            for cell in row:
                cell_str = str(cell).strip()
                if any(keyword in cell_str for keyword in location_keywords):
                    found_location_section = True
                    parts = re.split(r"[,:;\n]", cell_str)
                    for part in parts:
                        location = part.strip()
                        if location:
                            label = rule_based_location_extraction(location, known_states, known_districts, known_subdistricts, known_towns)
                            if label and final_filter(location, label):
                                locations.append((location, label))
                elif found_location_section:
                    parts = re.split(r"[,:;\n]", cell_str)
                    for part in parts:
                        location = part.strip()
                        if location:
                            label = rule_based_location_extraction(location, known_states, known_districts, known_subdistricts, known_towns)
                            if label and final_filter(location, label):
                                locations.append((location, label))
    return locations if locations else None

# Extract brief section from text
def extract_brief_section(text):
    match = re.search(r"Brief:\s*(.+?)(?=\n[A-Z][a-z]+:|\Z)", text, re.DOTALL) or \
            re.search(r"BRIEF\s*(.+?)(?=\n[A-Z][a-z]+:|\Z)", text, re.DOTALL)
    return match.group(1).strip() if match else None

# Final filtering function
def final_filter(loc, label):
    return (len(loc.split()) <= 3 and
            not re.match(r"^\d+$", loc) and
            not re.match(r"^\d+\.\d+$", loc) and
            not re.match(r"^\d+\.\d+\.\d+", loc) and
            not any(char in loc.lower() for char in ["%", "rs.", "cr.", "mu", "ha.", "govt", "ltd", "agreement", "pipeline", "site", "township", "tpp", "is", "o", "infrastructure", "scheme", "ca", "km", "has", "rail", "cr", "crore", "beneficiaries"]) and
            loc.lower() != "mp")

# Geocoding function
def get_coordinates_and_display_name(location, state=None):
    try:
        query = f"{location}, {state}, India" if state else f"{location}, India"
        
        # First, explicitly search only in India
        loc = geolocator.geocode(query, timeout=10, country_codes="IN")
        if loc:
            return (loc.latitude, loc.longitude, loc.raw.get('display_name', location))
        
        # Fallback: Try without state
        loc = geolocator.geocode(location, timeout=10, country_codes="IN")
        if loc:
            return (loc.latitude, loc.longitude, loc.raw.get('display_name', location))
        
        return None
    except GeocoderTimedOut:
        return None