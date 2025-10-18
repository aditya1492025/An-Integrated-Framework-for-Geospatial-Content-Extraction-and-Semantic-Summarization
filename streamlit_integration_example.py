"""
QUICK INTEGRATION EXAMPLE
========================
This shows exactly how to add performance metrics to your existing geo_interface.py

Just copy and paste these code snippets into your geo_interface.py file.
"""

# ============================================================================
# 1. ADD THESE IMPORTS AT THE TOP OF geo_interface.py
# ============================================================================

integration_imports = """
# Add these imports after your existing imports in geo_interface.py:

from performance_metrics import PipelineProfiler
import time
from datetime import datetime
"""

# ============================================================================
# 2. REPLACE YOUR FILE PROCESSING SECTION
# ============================================================================

streamlit_integration = """
# Replace the section "if uploaded_file and not st.session_state.processed:" 
# in your geo_interface.py with this instrumented version:

if uploaded_file and not st.session_state.processed:
    # Initialize performance profiler
    profiler = PipelineProfiler()
    profiler.start_pipeline()
    
    with st.spinner("Processing PDF and measuring performance..."):
        # Save temporary file
        tmp_file_path = f"temp_{uploaded_file.name}"
        with open(tmp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            # Time title extraction
            st.session_state.project_title = profiler.time_module(
                "TITLE_EXTRACTION", 
                extract_title, 
                tmp_file_path
            )
            
            # Time text extraction
            text = profiler.time_module(
                "TEXT_EXTRACTION", 
                extract_text, 
                tmp_file_path
            )
            
            # Time table extraction
            tables = profiler.time_module(
                "TABLE_EXTRACTION", 
                extract_tables, 
                tmp_file_path
            )
            
            # Time location processing
            def process_locations():
                brief_section = extract_brief_section(text)
                return process_text_for_locations(
                    brief_section or text[:2000], 
                    trained_nlp, untrained_nlp, 
                    known_states, known_districts, 
                    known_subdistricts, known_towns,
                    limit=10
                )
            
            locations_list = profiler.time_module(
                "NER_LOCATION_EXTRACTION",
                process_locations
            )
            
            # Time geocoding
            def geocode_and_create_dataframe(locations_list):
                locations_data = []
                successful_geocoding = 0
                total_attempts = 0
                
                for location, label in locations_list:
                    total_attempts += 1
                    coords_result = get_coordinates_and_display_name(location)
                    if coords_result and coords_result[0] is not None:
                        successful_geocoding += 1
                        lat, lon, display_name = coords_result
                        locations_data.append({
                            "Location": display_name,
                            "Type": label,
                            "Latitude": lat,
                            "Longitude": lon
                        })
                
                # Store geocoding stats for display
                st.session_state.geocoding_stats = {
                    'successful': successful_geocoding,
                    'total': total_attempts,
                    'success_rate': (successful_geocoding / total_attempts * 100) if total_attempts > 0 else 0
                }
                
                return pd.DataFrame(locations_data)
            
            st.session_state.locations_df = profiler.time_module(
                "GEOCODING",
                geocode_and_create_dataframe,
                locations_list
            )
            
            # Time web scraping (if enabled)
            if use_web_data:
                web_scraped_text, searched_urls, contributing_urls = profiler.time_module(
                    "WEB_SCRAPING",
                    scrape_web_data,
                    st.session_state.project_title
                )
                
                # Display web scraping info
                if contributing_urls:
                    st.info(f"Web data collected from {len(contributing_urls)} sources")
            else:
                web_scraped_text = ""
            
            # Time RAG processing
            st.session_state.rag_response = profiler.time_module(
                "RAG_ISSUES_ANALYSIS",
                query_rag,
                f"What are the key issues and challenges for the {st.session_state.project_title} project?",
                pdf_collection, web_collection if use_web_data else None, llm, st.session_state.project_title
            )
            
            st.session_state.ppt_response = profiler.time_module(
                "RAG_PPT_INSIGHTS",
                query_rag,
                f"Generate insights for {st.session_state.project_title}",
                pdf_collection, web_collection if use_web_data else None, llm, st.session_state.project_title,
                PPT_PROMPT_TEMPLATE
            )
            
            # End profiling and get results
            performance_results = profiler.end_pipeline()
            st.session_state.performance_metrics = performance_results
            
            # Save detailed results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_filename = f"results/performance_{timestamp}.json"
            os.makedirs("results", exist_ok=True)
            profiler.save_results(results_filename)
            
            st.session_state.processed = True
            st.success(f"Processing completed! Results saved to {results_filename}")
            
        except Exception as e:
            st.error(f"Processing failed: {str(e)}")
            profiler.end_pipeline()
        finally:
            # Cleanup temporary file
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
"""

# ============================================================================
# 3. ADD PERFORMANCE METRICS DISPLAY SECTION
# ============================================================================

performance_display = """
# Add this section after your results display in geo_interface.py:
# (Place it after the existing results but before the cleanup section)

# Performance Metrics Display
if st.session_state.processed and 'performance_metrics' in st.session_state:
    st.subheader("⚡ Performance Metrics")
    
    # Create columns for metrics display
    col1, col2, col3, col4 = st.columns(4)
    
    performance = st.session_state.performance_metrics
    total_time = performance.get("TOTAL_PIPELINE", 0)
    
    with col1:
        st.metric("Total Processing Time", f"{total_time:.2f}s")
    
    with col2:
        locations_count = len(st.session_state.locations_df) if not st.session_state.locations_df.empty else 0
        st.metric("Locations Found", locations_count)
    
    with col3:
        if 'geocoding_stats' in st.session_state:
            success_rate = st.session_state.geocoding_stats['success_rate']
            st.metric("Geocoding Success Rate", f"{success_rate:.1f}%")
        else:
            st.metric("Geocoding Success Rate", "N/A")
    
    with col4:
        # Calculate processing speed (locations per second)
        if total_time > 0 and locations_count > 0:
            speed = locations_count / total_time
            st.metric("Processing Speed", f"{speed:.1f} loc/s")
        else:
            st.metric("Processing Speed", "N/A")
    
    # Detailed breakdown in expandable section
    with st.expander("📊 Detailed Performance Breakdown"):
        
        # Module timing breakdown
        st.write("**Module Performance:**")
        
        module_times = {k: v for k, v in performance.items() if k != "TOTAL_PIPELINE"}
        
        # Create dataframe for better display
        breakdown_data = []
        for module, time_taken in sorted(module_times.items(), key=lambda x: x[1], reverse=True):
            percentage = (time_taken / total_time) * 100 if total_time > 0 else 0
            breakdown_data.append({
                "Module": module.replace("_", " ").title(),
                "Time (seconds)": f"{time_taken:.3f}",
                "Percentage": f"{percentage:.1f}%"
            })
        
        breakdown_df = pd.DataFrame(breakdown_data)
        st.dataframe(breakdown_df, use_container_width=True)
        
        # Performance insights
        st.write("**Performance Insights:**")
        
        if module_times:
            fastest_module = min(module_times.items(), key=lambda x: x[1])
            slowest_module = max(module_times.items(), key=lambda x: x[1])
            
            st.write(f"• Fastest operation: {fastest_module[0].replace('_', ' ').title()} ({fastest_module[1]:.3f}s)")
            st.write(f"• Slowest operation: {slowest_module[0].replace('_', ' ').title()} ({slowest_module[1]:.3f}s)")
            
            # Geocoding specific metrics
            if 'geocoding_stats' in st.session_state:
                stats = st.session_state.geocoding_stats
                st.write(f"• Geocoding: {stats['successful']}/{stats['total']} locations successfully geocoded")
                
                if stats['success_rate'] < 70:
                    st.warning("⚠️ Geocoding success rate below 70%. Check internet connection or location name quality.")
                elif stats['success_rate'] > 90:
                    st.success("✅ Excellent geocoding success rate!")
        
        # Memory and system info
        try:
            import psutil
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            st.write(f"• Current memory usage: {memory_mb:.1f} MB")
        except ImportError:
            st.write("• Install psutil for memory monitoring: `pip install psutil`")
        
        # Performance recommendations
        st.write("**Performance Tips:**")
        
        if total_time > 30:
            st.write("⏱️ Consider processing smaller document sections or using GPU acceleration")
        
        if 'RAG_ISSUES_ANALYSIS' in performance and performance['RAG_ISSUES_ANALYSIS'] > 15:
            st.write("🤖 RAG processing is slow. Consider using a smaller model or reducing context size")
        
        if 'GEOCODING' in performance and performance['GEOCODING'] > 10:
            st.write("🌐 Geocoding is slow. Consider caching results or using batch geocoding")
"""

# ============================================================================
# COMPLETE EXAMPLE USAGE
# ============================================================================

def create_complete_example():
    """Create a complete working example."""
    
    example_code = f"""
# COMPLETE INTEGRATION EXAMPLE
# ===========================
# 
# This is a complete example showing how to integrate performance 
# measurements into your existing geo_interface.py application.

{integration_imports}

# Add the performance-enabled processing logic:
{streamlit_integration}

# Add the performance metrics display:
{performance_display}

# That's it! Your app now includes comprehensive performance monitoring.
# 
# WHAT YOU'LL GET:
# - Real-time timing of all pipeline components
# - Geocoding success rate tracking  
# - Performance breakdown by module
# - Automatic results saving to JSON files
# - Visual performance metrics in the UI
# - Performance insights and recommendations
#
# USAGE:
# 1. Copy the import statements to the top of geo_interface.py
# 2. Replace your file processing section with the instrumented version
# 3. Add the performance display section where you want metrics shown
# 4. Run: streamlit run geo_interface.py
# 5. Upload a PDF and see performance metrics automatically!
"""
    
    return example_code

if __name__ == "__main__":
    print("🔗 STREAMLIT INTEGRATION GUIDE")
    print("==============================")
    print()
    print("This file shows exactly how to add performance metrics to your Streamlit app.")
    print()
    print("QUICK STEPS:")
    print("1. Copy the imports to geo_interface.py")
    print("2. Replace the file processing section") 
    print("3. Add the performance display section")
    print("4. Run your Streamlit app!")
    print()
    print(create_complete_example())