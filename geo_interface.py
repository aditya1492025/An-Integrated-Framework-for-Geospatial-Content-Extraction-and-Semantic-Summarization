import streamlit as st
import tempfile
import os
import pandas as pd
from geolocation import (
    load_models,
    load_predefined_locations,
    extract_text,
    extract_title,
    extract_tables,
    extract_location_from_tables,
    extract_brief_section,
    process_text_for_locations,
    get_coordinates_and_display_name
)
from rag_utils import (
    initialize_embedding_and_db,
    initialize_llm,
    chunk_text,
    query_rag,
    PROMPT_TEMPLATE,
    PPT_PROMPT_TEMPLATE,
    setup_rag_chain
)
from web_scraper import scrape_web_data
import streamlit.components.v1 as components

# Initialize resources only once
@st.cache_resource
def init_resources():
    trained_nlp, untrained_nlp = load_models()
    known_states, known_districts, known_subdistricts, known_towns = load_predefined_locations()
    embed_model, pdf_collection, web_collection = initialize_embedding_and_db()
    llm = initialize_llm()
    return trained_nlp, untrained_nlp, known_states, known_districts, known_subdistricts, known_towns, embed_model, pdf_collection, web_collection, llm

def render_map(locations_df):
    """Render the map with OpenLayers v2.13 using Bhuvan WMS."""
    if locations_df.empty:
        st.warning("No locations available to display on the map.")
        return None
    
    # Calculate map center and bounds
    center_lon = (locations_df["Longitude"].min() + locations_df["Longitude"].max()) / 2
    center_lat = (locations_df["Latitude"].min() + locations_df["Latitude"].max()) / 2
    bounds = [
        locations_df["Longitude"].min(), locations_df["Latitude"].min(),
        locations_df["Longitude"].max(), locations_df["Latitude"].max()
    ]

    # Generate markers as OpenLayers features
    markers_js = ""
    for idx, row in locations_df.iterrows():
        markers_js += f"""
        new OpenLayers.Feature.Vector(
            new OpenLayers.Geometry.Point({row['Longitude']}, {row['Latitude']}).transform(
                new OpenLayers.Projection("EPSG:4326"),
                map.getProjectionObject()
            ),
            {{name: '{row['Display Name']}<br>Lat: {row['Latitude']:.6f}<br>Lon: {row['Longitude']:.6f}'}}
        ),
        """

    # HTML and JS for OpenLayers map (v2.13 syntax)
    map_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/openlayers/2.13.1/OpenLayers.js"></script>
        <style>
            #map {{
                width: 100%;
                height: 500px;
            }}
        </style>
    </head>
    <body>
        <div id="map"></div>
        <script>
            var map = new OpenLayers.Map("map");
            
            // Bhuvan WMS layer
            var wms = new OpenLayers.Layer.WMS(
                "Bhuvan Base MAP",
                "https://bhuvan-vec1.nrsc.gov.in/bhuvan/gwc/service/wms",
                {{layers: "india3", format: "image/png"}},
                {{isBaseLayer: true}}
            );
            map.addLayer(wms);

            // Bhuvan WMS layer
            var wms1 = new OpenLayers.Layer.WMS(
                "Bhuvan Base Satellite",
                "https://bhuvan-ras1.nrsc.gov.in/tilecache/tilecache.py",
                {{layers: "bhuvan_img", format: "image/jpeg"}},
                {{isBaseLayer: true}}
            );
            map.addLayer(wms1);

            // Markers layer
            var markers = new OpenLayers.Layer.Vector("Markers", {{
                styleMap: new OpenLayers.StyleMap({{
                    "default": {{
                        externalGraphic: "https://docs.maptiler.com/openlayers/examples/default-marker/marker-icon.png",
                        graphicWidth: 25,
                        graphicHeight: 41,
                        graphicYOffset: -41
                    }}
                }})
            }});
            markers.addFeatures([
                {markers_js[:-2]}  // Remove trailing comma
            ]);
            map.addLayer(markers);


            // Add the Layer Switcher control
            map.addControl(new OpenLayers.Control.LayerSwitcher());

            // Popup on click
            var selectCtrl = new OpenLayers.Control.SelectFeature(markers, {{
                onSelect: function(feature) {{
                    var popup = new OpenLayers.Popup.FramedCloud(
                        "popup",
                        feature.geometry.getBounds().getCenterLonLat(),
                        null,
                        feature.attributes.name,
                        null,
                        true,
                        function() {{ selectCtrl.unselectAll(); }}
                    );
                    feature.popup = popup;
                    map.addPopup(popup);
                }},
                onUnselect: function(feature) {{
                    if (feature.popup) {{
                        map.removePopup(feature.popup);
                        feature.popup.destroy();
                        feature.popup = null;
                    }}
                }}
            }});
            map.addControl(selectCtrl);
            selectCtrl.activate();

            // Set center and zoom to extent
            var bounds = new OpenLayers.Bounds({bounds[0]}, {bounds[1]}, {bounds[2]}, {bounds[3]}).transform(
                new OpenLayers.Projection("EPSG:4326"),
                map.getProjectionObject()
            );
            map.zoomToExtent(bounds);
            if (map.getZoom() > 10) map.zoomTo(10);  // Limit max zoom
        </script>
    </body>
    </html>
    """
    
    # Render map in Streamlit
    components.html(map_html, height=510)
    #st.info(f"Map extent: BBOX={bounds}")

# Streamlit UI
# Create two columns
col1, col2 = st.columns(2)

# Display logo in the first column
with col1:
    st.image("bhuvan logo.webp", width=200)  

# Display logo in the second column
with col2:
    st.image("nrsc_logo_412023_new.png", width=250)  
st.title("An Integrated Framework for Geospatial Content Extraction and Semantic Summarization from PDF Documents")
st.markdown("Upload a PDF to extract geographic locations, process text with RAG, and visualize project locations with Bhuvan Maps")

# Toggle for using web-scraped data
use_web_data = st.checkbox("Use web-scraped data for RAG analysis", value=True, help="Uncheck to exclude web data from RAG analysis.")

# Initialize resources
trained_nlp, untrained_nlp, known_states, known_districts, known_subdistricts, known_towns, embed_model, pdf_collection, web_collection, llm = init_resources()

# Initialize session state variables
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'locations_df' not in st.session_state:
    st.session_state.locations_df = pd.DataFrame()
if 'project_title' not in st.session_state:
    st.session_state.project_title = ""
if 'rag_response' not in st.session_state:
    st.session_state.rag_response = ""
if 'ppt_response' not in st.session_state:
    st.session_state.ppt_response = ""

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file and not st.session_state.processed:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        if st.button("Process PDF"):
            with st.spinner("Analyzing document..."):
                full_text = extract_text(tmp_file_path)
                project_title = extract_title(tmp_file_path) or "Untitled Project"
                tables = extract_tables(tmp_file_path)

                text_chunks = chunk_text(full_text, chunk_size=200)
                if pdf_collection is not None:
                    pdf_collection.add_texts(texts=text_chunks, ids=[f"pdf_chunk_{i}" for i in range(len(text_chunks))])
                    st.success(f"Stored {len(text_chunks)} PDF chunks in ChromaDB")
                    #st.write(project_title)
                else:
                    st.error("PDF collection initialization failed.")

                if use_web_data and web_collection is not None:
                    web_data, searched_urls, contributing_urls = scrape_web_data(project_title)
                    web_chunks = chunk_text(web_data, chunk_size=200)
                    web_collection.add_texts(texts=web_chunks, ids=[f"web_chunk_{i}" for i in range(len(web_chunks))])
                    st.success(f"Stored {len(web_chunks)} web chunks in ChromaDB")
                    # Display URLs
                    st.subheader("Web Scraping Details")
                    st.write(f"**Searched URLs**: {', '.join(searched_urls)}")
                    st.write(f"**Contributing URLs**: {', '.join(contributing_urls) if contributing_urls else 'None'}")
                elif not use_web_data:
                    st.info("Web-scraped data excluded from RAG analysis as per user selection.")

                all_locations = set()
                if project_title:
                    title_locations = process_text_for_locations(project_title, trained_nlp, untrained_nlp, known_states, known_districts, known_subdistricts, known_towns)
                    all_locations.update(title_locations)

                table_locations = extract_location_from_tables(tables, known_states, known_districts, known_subdistricts, known_towns)
                if table_locations:
                    all_locations.update(table_locations)

                brief_section = extract_brief_section(full_text)
                if brief_section:
                    brief_limited_locations = process_text_for_locations(brief_section, trained_nlp, untrained_nlp, known_states, known_districts, known_subdistricts, known_towns, limit=10)
                    all_locations.update(brief_limited_locations)

                if all_locations:
                    sorted_locations = sorted(all_locations, key=lambda x: (["STATE", "DISTRICT", "SUBDISTRICT", "TOWN"].index(x[1]), x[0]))
                    limited_locations = sorted_locations[:10]
                    state = next((loc for loc, label in limited_locations if label == "STATE"), None)
                    final_locations = []
                    
                    for loc, label in limited_locations:
                        result = get_coordinates_and_display_name(loc, state)
                        if result:
                            lat, lon, display_name = result
                            final_locations.append({"Location": loc, "Type": label, "Latitude": float(lat), "Longitude": float(lon), "Display Name": display_name})

                    if final_locations:
                        df = pd.DataFrame(final_locations)
                        df["Latitude"] = df["Latitude"].astype('float64')
                        df["Longitude"] = df["Longitude"].astype('float64')
                        pd.set_option('display.float_format', '{:.15f}'.format)
                        st.session_state.locations_df = df
                    else:
                        st.warning("No valid locations found with coordinates")
                else:
                    st.warning("No locations found in the document")

                if pdf_collection is not None and llm is not None:
                    st.session_state.project_title = project_title
                    if use_web_data and web_collection is not None:
                        chain = setup_rag_chain(llm, pdf_collection, web_collection, PROMPT_TEMPLATE)
                        ppt_chain = setup_rag_chain(llm, pdf_collection, web_collection, PPT_PROMPT_TEMPLATE)
                    else:
                        chain = setup_rag_chain(llm, pdf_collection, None, PROMPT_TEMPLATE)
                        ppt_chain = setup_rag_chain(llm, pdf_collection, None, PPT_PROMPT_TEMPLATE)
                    
                    query = "What are the most urgent issues in the document?"
                    st.session_state.rag_response = chain({"query": query})["result"].strip()
                    st.session_state.ppt_response = ppt_chain({"query": "What are key insights from the document for a presentation?"})["result"].strip()
                    st.success("RAG query processed!")
                else:
                    st.warning("RAG analysis skipped due to ChromaDB or LLM initialization failure.")

                st.session_state.processed = True
                st.session_state.tmp_file_path = tmp_file_path

    except Exception as e:
        st.error(f"An error occurred: {e}")
        if 'tmp_file_path' in st.session_state:
            os.unlink(st.session_state.tmp_file_path)
            del st.session_state.tmp_file_path

# Display results if processed
if st.session_state.processed:
    st.subheader("Final Project Locations")
    if not st.session_state.locations_df.empty:
        st.dataframe(st.session_state.locations_df[["Location", "Type", "Latitude", "Longitude", "Display Name"]])

        locations_to_remove = st.multiselect(
            "Select irrelevant locations to remove",
            options=st.session_state.locations_df["Display Name"].tolist(),
            default=[],
            key="location_selector"
        )

        if st.button("Update Map"):
            if locations_to_remove:
                st.session_state.locations_df = st.session_state.locations_df[
                    ~st.session_state.locations_df["Display Name"].isin(locations_to_remove)
                ]
                st.success(f"Removed {len(locations_to_remove)} location(s) from the map.")

        st.subheader("Geographic Visualization (Bhuvan Maps)")
        render_map(st.session_state.locations_df)
        st.markdown("Note: Click markers to see location names and coordinates.")
        
        csv = st.session_state.locations_df.to_csv(index=False, float_format='%.15f').encode('utf-8')
        st.download_button(label="Download updated locations as CSV", data=csv, file_name="updated_locations.csv", mime="text/csv")
    else:
        st.warning("No valid locations to display.")

    if st.session_state.rag_response or st.session_state.ppt_response:
        st.subheader("RAG Analysis")
        st.text_area("Urgent Issues and Solutions", st.session_state.rag_response, height=200)
        st.subheader("Key Insights for Presentation")
        st.text_area("PPT Points", st.session_state.ppt_response, height=200)

# Cleanup temporary file after processing
if 'tmp_file_path' in st.session_state and st.session_state.processed:
    os.unlink(st.session_state.tmp_file_path)
    del st.session_state.tmp_file_path