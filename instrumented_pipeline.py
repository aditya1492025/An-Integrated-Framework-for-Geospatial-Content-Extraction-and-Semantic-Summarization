"""
Instrumented Pipeline for Performance Evaluation
Integrates with existing geolocation.py, rag_utils.py, and web_scraper.py modules
"""

import streamlit as st
import time
from datetime import datetime
from performance_metrics import PipelineProfiler, measure_memory_usage

# Import your existing modules
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

def instrumented_pdf_extraction_module(uploaded_file, tmp_file_path):
    """
    Instrumented version of PDF extraction module.
    Measures time for text extraction, title extraction, and table extraction.
    """
    profiler = PipelineProfiler()
    results = {}
    
    print("Starting PDF Extraction Module...")
    module_start = time.perf_counter()
    
    # Extract text
    with st.spinner("Extracting text from PDF..."):
        text_start = time.perf_counter()
        text = extract_text(tmp_file_path)
        text_time = time.perf_counter() - text_start
        results['text_extraction_time'] = text_time
        print(f"  Text extraction: {text_time:.3f} seconds")
    
    # Extract title
    title_start = time.perf_counter()
    project_title = extract_title(tmp_file_path)
    title_time = time.perf_counter() - title_start
    results['title_extraction_time'] = title_time
    print(f"  Title extraction: {title_time:.3f} seconds")
    
    # Extract tables
    table_start = time.perf_counter()
    tables = extract_tables(tmp_file_path)
    table_time = time.perf_counter() - table_start
    results['table_extraction_time'] = table_time
    print(f"  Table extraction: {table_time:.3f} seconds")
    
    module_total = time.perf_counter() - module_start
    results['total_module_time'] = module_total
    
    print(f"✓ PDF Extraction Module completed: {module_total:.3f} seconds")
    
    return {
        'text': text,
        'project_title': project_title,
        'tables': tables,
        'performance': results
    }

def instrumented_ner_and_geocoding_module(text, tables, trained_nlp, untrained_nlp, 
                                        known_states, known_districts, 
                                        known_subdistricts, known_towns):
    """
    Instrumented version of NER and geocoding module.
    Measures time for location extraction, geocoding, and validation.
    """
    results = {}
    print("Starting NER and Geocoding Module...")
    module_start = time.perf_counter()
    
    # Extract brief section
    brief_start = time.perf_counter()
    brief_section = extract_brief_section(text)
    brief_time = time.perf_counter() - brief_start
    results['brief_extraction_time'] = brief_time
    print(f"  Brief section extraction: {brief_time:.3f} seconds")
    
    # Process locations from text
    location_start = time.perf_counter()
    text_locations = process_text_for_locations(
        brief_section or text[:2000], 
        trained_nlp, untrained_nlp, 
        known_states, known_districts, 
        known_subdistricts, known_towns, 
        limit=10
    )
    location_time = time.perf_counter() - location_start
    results['text_location_processing_time'] = location_time
    print(f"  Text location processing: {location_time:.3f} seconds")
    
    # Extract locations from tables
    table_location_start = time.perf_counter()
    table_locations = extract_location_from_tables(
        tables, known_states, known_districts, 
        known_subdistricts, known_towns
    )
    table_location_time = time.perf_counter() - table_location_start
    results['table_location_processing_time'] = table_location_time
    print(f"  Table location processing: {table_location_time:.3f} seconds")
    
    # Combine and geocode locations
    geocoding_start = time.perf_counter()
    all_locations = text_locations + (table_locations or [])
    locations_df = []
    
    geocoding_count = 0
    successful_geocoding = 0
    
    for location, label in all_locations[:10]:  # Limit for performance
        geocoding_count += 1
        coords_result = get_coordinates_and_display_name(location)
        if coords_result and coords_result[0] is not None:
            successful_geocoding += 1
            lat, lon, display_name = coords_result
            locations_df.append({
                "Location": display_name,
                "Type": label,
                "Latitude": lat,
                "Longitude": lon
            })
    
    geocoding_time = time.perf_counter() - geocoding_start
    results['geocoding_time'] = geocoding_time
    results['geocoding_count'] = geocoding_count
    results['successful_geocoding'] = successful_geocoding
    results['geocoding_success_rate'] = (successful_geocoding / geocoding_count * 100) if geocoding_count > 0 else 0
    
    print(f"  Geocoding ({geocoding_count} locations): {geocoding_time:.3f} seconds")
    print(f"  Geocoding success rate: {results['geocoding_success_rate']:.1f}%")
    
    module_total = time.perf_counter() - module_start
    results['total_module_time'] = module_total
    
    print(f"✓ NER and Geocoding Module completed: {module_total:.3f} seconds")
    
    return {
        'locations_df': locations_df,
        'extracted_locations': all_locations,
        'performance': results
    }

def instrumented_embedding_and_storage_module(text, web_scraped_text, embed_model):
    """
    Instrumented version of embedding and storage module.
    Measures time for text chunking, embedding generation, and vector storage.
    """
    results = {}
    print("Starting Embedding and Storage Module...")
    module_start = time.perf_counter()
    
    # Initialize collections
    init_start = time.perf_counter()
    embed_model, pdf_collection, web_collection = initialize_embedding_and_db()
    init_time = time.perf_counter() - init_start
    results['db_initialization_time'] = init_time
    print(f"  Database initialization: {init_time:.3f} seconds")
    
    # Chunk and store PDF text
    pdf_chunk_start = time.perf_counter()
    pdf_chunks = chunk_text(text, chunk_size=200)
    
    # Store PDF chunks (simplified - replace with your actual storage logic)
    for i, chunk in enumerate(pdf_chunks[:50]):  # Limit for performance testing
        pdf_collection.add_texts(
            texts=[chunk],
            metadatas=[{"source": "pdf", "chunk_id": i}],
            ids=[f"pdf_chunk_{i}"]
        )
    
    pdf_chunk_time = time.perf_counter() - pdf_chunk_start
    results['pdf_chunking_storage_time'] = pdf_chunk_time
    results['pdf_chunks_count'] = len(pdf_chunks[:50])
    print(f"  PDF chunking and storage: {pdf_chunk_time:.3f} seconds ({results['pdf_chunks_count']} chunks)")
    
    # Chunk and store web text
    if web_scraped_text:
        web_chunk_start = time.perf_counter()
        web_chunks = chunk_text(web_scraped_text, chunk_size=200)
        
        for i, chunk in enumerate(web_chunks[:20]):  # Limit for performance
            web_collection.add_texts(
                texts=[chunk],
                metadatas=[{"source": "web", "chunk_id": i}],
                ids=[f"web_chunk_{i}"]
            )
        
        web_chunk_time = time.perf_counter() - web_chunk_start
        results['web_chunking_storage_time'] = web_chunk_time
        results['web_chunks_count'] = len(web_chunks[:20])
        print(f"  Web chunking and storage: {web_chunk_time:.3f} seconds ({results['web_chunks_count']} chunks)")
    else:
        results['web_chunking_storage_time'] = 0
        results['web_chunks_count'] = 0
    
    module_total = time.perf_counter() - module_start
    results['total_module_time'] = module_total
    
    print(f"✓ Embedding and Storage Module completed: {module_total:.3f} seconds")
    
    return {
        'pdf_collection': pdf_collection,
        'web_collection': web_collection,
        'performance': results
    }

def instrumented_rag_summarization_module(project_title, pdf_collection, web_collection, llm):
    """
    Instrumented version of RAG summarization module.
    Measures time for query processing, retrieval, and generation.
    """
    results = {}
    print("Starting RAG Summarization Module...")
    module_start = time.perf_counter()
    
    # Generate RAG response
    rag_start = time.perf_counter()
    rag_query = f"What are the key issues and challenges for the {project_title} project?"
    rag_response = query_rag(rag_query, pdf_collection, web_collection, llm, project_title)
    rag_time = time.perf_counter() - rag_start
    results['rag_query_time'] = rag_time
    print(f"  RAG query processing: {rag_time:.3f} seconds")
    
    # Generate PPT insights
    ppt_start = time.perf_counter()
    ppt_response = query_rag(
        f"Generate insights for {project_title}", 
        pdf_collection, web_collection, llm, project_title, 
        PPT_PROMPT_TEMPLATE
    )
    ppt_time = time.perf_counter() - ppt_start
    results['ppt_generation_time'] = ppt_time
    print(f"  PPT insights generation: {ppt_time:.3f} seconds")
    
    module_total = time.perf_counter() - module_start
    results['total_module_time'] = module_total
    
    print(f"✓ RAG Summarization Module completed: {module_total:.3f} seconds")
    
    return {
        'rag_response': rag_response,
        'ppt_response': ppt_response,
        'performance': results
    }

def run_instrumented_pipeline(uploaded_file):
    """
    Main instrumented pipeline that measures comprehensive performance metrics.
    """
    # Initialize main profiler
    main_profiler = PipelineProfiler()
    main_profiler.start_pipeline()
    
    # Initialize resources
    print("Initializing resources...")
    init_start = time.perf_counter()
    
    trained_nlp, untrained_nlp, known_states, known_districts, known_subdistricts, known_towns, embed_model, pdf_collection, web_collection, llm = init_resources()
    
    init_time = time.perf_counter() - init_start
    print(f"Resource initialization: {init_time:.3f} seconds")
    
    # Save uploaded file temporarily
    tmp_file_path = f"temp_{uploaded_file.name}"
    with open(tmp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    try:
        # Run each instrumented module
        pdf_results = main_profiler.time_module(
            "PDF_EXTRACTION",
            instrumented_pdf_extraction_module,
            uploaded_file, tmp_file_path
        )
        
        ner_results = main_profiler.time_module(
            "NER_AND_GEOCODING",
            instrumented_ner_and_geocoding_module,
            pdf_results['text'], pdf_results['tables'],
            trained_nlp, untrained_nlp, known_states, 
            known_districts, known_subdistricts, known_towns
        )
        
        # Web scraping
        web_start = time.perf_counter()
        web_scraped_text, _, _ = scrape_web_data(pdf_results['project_title'])
        web_time = time.perf_counter() - web_start
        print(f"Web scraping: {web_time:.3f} seconds")
        
        embedding_results = main_profiler.time_module(
            "EMBEDDING_AND_STORAGE",
            instrumented_embedding_and_storage_module,
            pdf_results['text'], web_scraped_text, embed_model
        )
        
        rag_results = main_profiler.time_module(
            "RAG_SUMMARIZATION",
            instrumented_rag_summarization_module,
            pdf_results['project_title'],
            embedding_results['pdf_collection'],
            embedding_results['web_collection'],
            llm
        )
        
        # Generate final report
        final_timings = main_profiler.end_pipeline()
        
        # Compile comprehensive results
        comprehensive_results = {
            'pipeline_timings': final_timings,
            'pdf_performance': pdf_results.get('performance', {}),
            'ner_performance': ner_results.get('performance', {}),
            'embedding_performance': embedding_results.get('performance', {}),
            'rag_performance': rag_results.get('performance', {}),
            'memory_usage': measure_memory_usage(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save detailed results
        main_profiler.save_results("comprehensive_performance.json")
        
        return {
            'locations_df': ner_results['locations_df'],
            'project_title': pdf_results['project_title'],
            'rag_response': rag_results['rag_response'],
            'ppt_response': rag_results['ppt_response'],
            'performance_metrics': comprehensive_results
        }
        
    finally:
        # Cleanup
        import os
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

# Initialize resources function (you'll need to adapt this to your init_resources function)
@st.cache_resource
def init_resources():
    """Initialize all required resources - adapt this to match your existing function."""
    # This should match your existing init_resources() function from geo_interface.py
    trained_nlp, untrained_nlp = load_models()
    known_states, known_districts, known_subdistricts, known_towns = load_predefined_locations()
    embed_model, pdf_collection, web_collection = initialize_embedding_and_db()
    llm = initialize_llm()
    return trained_nlp, untrained_nlp, known_states, known_districts, known_subdistricts, known_towns, embed_model, pdf_collection, web_collection, llm

if __name__ == "__main__":
    # This can be used for standalone testing
    print("Performance measurement module ready.")
    print("Use run_instrumented_pipeline() with your Streamlit app.")