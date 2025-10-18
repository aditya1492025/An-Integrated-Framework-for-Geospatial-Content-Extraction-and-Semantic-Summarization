# An Integrated Framework for Geospatial Content Extraction and Semantic Summarization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![spaCy](https://img.shields.io/badge/spaCy-3.7+-green.svg)](https://spacy.io/)
[![License](https://img.shields.io/badge/license-Research%20Only-red.svg)](#)

> **Research Framework** for automated geospatial content extraction and semantic summarization from PDF documents using advanced NLP and RAG techniques.

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Basic Usage
```python
# Extract and geocode locations from PDF
from geo_interface import GeospatialProcessor
from rag_utils import RAGProcessor

processor = GeospatialProcessor()
locations = processor.extract_locations("document.pdf")

# Generate semantic summary
rag = RAGProcessor()
summary = rag.generate_summary(locations, "document.pdf")
```

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PDF Input     │───▶│  NER Processing  │───▶│   Geocoding     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Semantic Summary│◀───│  RAG Processing  │◀───│  Location Data  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📊 Performance Overview

| **Metric** | **Value** | **Performance Level** |
|------------|-----------|----------------------|
| Processing Speed | 0.80 locations/sec | ⚡ Real-time |
| Entity Recognition F1 | 0.722 | 🎯 Good |
| Geocoding Success | 100% | ✅ Perfect |
| RAG Quality Score | 4.14/5.0 | 🌟 Excellent |
| Overall Accuracy | 79.6% | ✅ Production Ready |

## 🔧 Core Components

### 1. **PDF Processing** (`web_scraper.py`)
- Automated text extraction from PDF documents
- Content preprocessing and structure analysis

### 2. **Named Entity Recognition** (`geolocation.py`)
- Custom spaCy transformer model for geographical entities
- 634,163 predefined Indian locations database

### 3. **Geocoding Engine** (`geo_interface.py`)
- Nominatim-based coordinate resolution
- Multi-level fallback strategies for accuracy

### 4. **RAG System** (`rag_utils.py`)
- ChromaDB vector storage and retrieval
- Mistral-7B-Instruct semantic generation

### 5. **Evaluation Framework** (`performance_metrics.py`)
- Comprehensive performance monitoring
- Real-time metrics collection and analysis

## 📈 Evaluation Results

### 📊 Complete Performance Documentation

- **[PERFORMANCE_METRICS.md](PERFORMANCE_METRICS.md)** - Detailed evaluation methodology and analysis
- **[ALL_METRICS_TABLES.md](ALL_METRICS_TABLES.md)** - Ready-to-use tables for papers and presentations

**Key Results Summary:**
- ⚡ **Processing Speed**: 0.80 locations/second (real-time capability)
- 🎯 **Perfect Precision**: 1.000 (zero false positives) 
- ✅ **Geocoding Success**: 100% (reliable coordinate assignment)
- 🌟 **RAG Quality**: 4.14/5.0 (expert-evaluated excellence)
- 📊 **Overall Accuracy**: 79.6% (production deployment ready)

## 🛠️ Installation & Setup

1. **Clone Repository**
```bash
git clone <repository-url>
cd An-Integrated-Framework-for-Geospatial-Content-Extraction-and-Semantic-Summarization
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Download Models**
```bash
# spaCy model (included in Trained_spacy_model/)
# Pre-trained embeddings will be downloaded automatically
```

4. **Run Evaluation**
```bash
python comprehensive_evaluation.py
```

## 📁 Repository Structure

```
├── 📄 Core Framework
│   ├── geo_interface.py      # Main geospatial processing interface
│   ├── geolocation.py        # NER and entity extraction
│   ├── rag_utils.py         # RAG processing and summarization
│   └── web_scraper.py       # PDF content extraction
├── 🧪 Evaluation Suite
│   ├── comprehensive_evaluation.py    # Full framework evaluation
│   ├── detailed_accuracy_evaluation.py # Accuracy metrics
│   ├── performance_metrics.py         # Performance monitoring
│   └── final_results_summary.py       # Results compilation
├── 🤖 Models & Data
│   ├── Trained_spacy_model/          # Custom NER model
│   ├── chroma_db/                    # Vector database
│   └── predefined_locations.csv      # Geographical entities
└── 📊 Documentation
    ├── PERFORMANCE_METRICS.md        # Detailed evaluation results
    └── evaluation_methodology.md     # Assessment methodology
```

## 🎯 Key Features

- **� Real-time Processing**: 0.80 locations/second processing speed
- **🎯 High Precision**: Perfect precision (1.000) in entity recognition
- **✅ Reliable Geocoding**: 100% success rate for coordinate assignment
- **🌟 Quality Summarization**: 4.14/5.0 expert-evaluated RAG quality
- **🔧 Modular Design**: Scalable architecture for easy customization
- **📊 Comprehensive Evaluation**: Rigorous performance assessment framework

## 🔬 Research Applications

- **📋 Project Documentation Analysis**: Automated extraction from infrastructure reports
- **🗺️ Geographic Information Systems**: Integration with GIS workflows  
- **📊 Decision Support Systems**: Semantic summarization for stakeholders
- **🔍 Content Discovery**: Large-scale document analysis and indexing

## 💻 System Requirements & Validation

### Evaluation System Specifications
Our framework was evaluated on:
- **System**: HP Victus Gaming Laptop 15-fa0xxx
- **CPU**: Intel Core i5-12450H (8 cores, 12 threads @ 2.0GHz)
- **Memory**: 16 GB RAM
- **GPU**: NVIDIA GeForce GTX 1650 (4GB VRAM)
- **Storage**: 512GB NVMe SSD
- **OS**: Windows 11 Home Single Language

### Performance Validation
✅ **Real-time processing** achieved on mid-range gaming laptop  
✅ **4-6 GB RAM usage** - fits comfortably in 16GB system  
✅ **Sub-second response times** - excellent user experience  
✅ **CUDA acceleration** - optimal GPU utilization  

## ⚠️ Usage Notice

This framework is developed for research purposes. The evaluation demonstrates excellent performance on standard gaming/business laptops. For production deployment, ensure appropriate computational resources and validate performance with domain-specific datasets.

## 📞 Contact

For questions about implementation or research collaboration, please refer to the paper submission details.

---

**Citation**: *Details will be provided upon paper acceptance*

If you are not an IEEE reviewer but wish to view, use, or cite this work, you must obtain **explicit written permission** from the authors.

For permissions or inquiries, please contact:

**Aditya Siras**  
📧 adityasiras@gmail.com

---

## 📎 Notes

- This repository may be removed or made private after the peer review process ends.
- Please respect the boundaries of the academic review process and intellectual ownership.
