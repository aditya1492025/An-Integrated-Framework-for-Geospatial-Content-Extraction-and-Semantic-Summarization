"""
Performance Metrics Module for Geospatial Content Extraction Framework
Author: Research Team
Date: October 2025

This module provides timing and performance measurement utilities for the 
integrated geospatial framework pipeline evaluation.
"""

import time
import functools
from typing import Dict, List, Callable, Any
from datetime import datetime
import json
import os

class PerformanceTimer:
    """Context manager and decorator for measuring execution times."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.duration = None
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
        
    def __call__(self, func: Callable) -> Callable:
        """Use as decorator to time function execution."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with PerformanceTimer(f"{func.__name__}") as timer:
                result = func(*args, **kwargs)
            return result, timer.duration
        return wrapper

class PipelineProfiler:
    """Comprehensive profiler for the entire geospatial pipeline."""
    
    def __init__(self):
        self.timings: Dict[str, float] = {}
        self.start_time = None
        self.memory_usage = {}
        
    def start_pipeline(self):
        """Start timing the entire pipeline."""
        self.start_time = time.perf_counter()
        print(f"\n{'='*60}")
        print(f"GEOSPATIAL FRAMEWORK PERFORMANCE EVALUATION")
        print(f"{'='*60}")
        print(f"Pipeline started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
        
    def time_module(self, module_name: str, module_func: Callable, *args, **kwargs):
        """Time a specific module and store results."""
        print(f"Starting {module_name}...")
        start_time = time.perf_counter()
        
        try:
            result = module_func(*args, **kwargs)
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            self.timings[module_name] = duration
            print(f"✓ {module_name} completed: {duration:.3f} seconds")
            return result
            
        except Exception as e:
            end_time = time.perf_counter()
            duration = end_time - start_time
            self.timings[f"{module_name}_FAILED"] = duration
            print(f"✗ {module_name} failed after {duration:.3f} seconds: {str(e)}")
            raise
            
    def end_pipeline(self):
        """End pipeline timing and generate comprehensive report."""
        total_time = time.perf_counter() - self.start_time
        self.timings["TOTAL_PIPELINE"] = total_time
        
        print(f"\n{'='*60}")
        print(f"PERFORMANCE EVALUATION RESULTS")
        print(f"{'='*60}")
        
        # Individual module timings
        print(f"\nModule Performance Breakdown:")
        print(f"{'-'*40}")
        
        module_times = {k: v for k, v in self.timings.items() if k != "TOTAL_PIPELINE"}
        for module, duration in module_times.items():
            percentage = (duration / total_time) * 100
            print(f"{module:<30}: {duration:>8.3f}s ({percentage:>5.1f}%)")
            
        print(f"{'-'*40}")
        print(f"{'TOTAL PIPELINE TIME':<30}: {total_time:>8.3f}s (100.0%)")
        
        # Performance summary
        print(f"\nPerformance Summary:")
        print(f"{'-'*40}")
        print(f"Fastest Module: {min(module_times, key=module_times.get)} ({min(module_times.values()):.3f}s)")
        print(f"Slowest Module: {max(module_times, key=module_times.get)} ({max(module_times.values()):.3f}s)")
        print(f"Average Module Time: {sum(module_times.values())/len(module_times):.3f}s")
        
        print(f"\nPipeline completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
        
        return self.timings
        
    def save_results(self, filename: str = "performance_results.json"):
        """Save timing results to JSON file for analysis."""
        results = {
            "timestamp": datetime.now().isoformat(),
            "timings": self.timings,
            "total_modules": len([k for k in self.timings.keys() if k != "TOTAL_PIPELINE"]),
            "success_rate": len([k for k in self.timings.keys() if "FAILED" not in k]) / len(self.timings) * 100
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Performance results saved to: {filename}")

def measure_memory_usage():
    """Measure current memory usage (requires psutil)."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024
        }
    except ImportError:
        print("Warning: psutil not installed. Memory monitoring unavailable.")
        return {"rss_mb": 0, "vms_mb": 0}

# Example usage functions that simulate your pipeline modules
def simulate_pdf_extraction_module(pdf_path: str):
    """Simulate PDF extraction with timing."""
    # Replace with your actual PDF extraction logic
    time.sleep(0.5)  # Simulated processing time
    return f"Extracted text from {pdf_path}"

def simulate_ner_and_geocoding_module(text: str):
    """Simulate NER and geocoding with timing."""
    # Replace with your actual NER/geocoding logic
    time.sleep(1.2)  # Simulated processing time
    return ["Location1", "Location2", "Location3"]

def simulate_embedding_and_storage_module(locations: List[str]):
    """Simulate embedding and storage with timing."""
    # Replace with your actual embedding/storage logic
    time.sleep(0.8)  # Simulated processing time
    return "Embeddings stored successfully"

def simulate_rag_summarization_module(query: str):
    """Simulate RAG summarization with timing."""
    # Replace with your actual RAG logic
    time.sleep(2.0)  # Simulated processing time
    return "Generated summary of the document"

if __name__ == "__main__":
    # Example usage of the performance measurement system
    profiler = PipelineProfiler()
    
    # Start pipeline timing
    profiler.start_pipeline()
    
    try:
        # Time each module
        pdf_text = profiler.time_module(
            "PDF_EXTRACTION", 
            simulate_pdf_extraction_module, 
            "sample.pdf"
        )
        
        locations = profiler.time_module(
            "NER_AND_GEOCODING", 
            simulate_ner_and_geocoding_module, 
            pdf_text
        )
        
        storage_result = profiler.time_module(
            "EMBEDDING_AND_STORAGE", 
            simulate_embedding_and_storage_module, 
            locations
        )
        
        summary = profiler.time_module(
            "RAG_SUMMARIZATION", 
            simulate_rag_summarization_module, 
            "What are the key insights?"
        )
        
        # End pipeline and generate report
        final_timings = profiler.end_pipeline()
        
        # Save results
        profiler.save_results("pipeline_performance.json")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        profiler.end_pipeline()