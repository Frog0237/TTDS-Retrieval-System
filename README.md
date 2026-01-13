# Information Retrieval System (Coursework Project)

This project implements a basic Information Retrieval (IR) system for academic coursework.  
The system processes a collection of XML documents and supports multiple query types through a positional inverted index.

## Implemented Features

### Document Processing
- Parses XML documents containing document IDs, headlines, and text content
- Performs text preprocessing including:
  - Tokenisation
  - Stopword removal
  - Normalisation
  - Porter stemming
- Outputs preprocessed documents with term positions

### Index Construction
- Builds a **positional inverted index**
- Records:
  - Term occurrences and positions per document
  - Document frequency for each term
  - Document length statistics
- Stores the index in a text-based format for later retrieval

### Query Processing
The system supports four types of search:

- **Boolean Search**
  - Operators: `AND`, `OR`, `NOT`
  - Parentheses for grouping
- **Phrase Search**
  - Exact phrase matching using positional information
- **Proximity Search**
  - Queries of the form `#k(term1, term2)`
- **Ranked Retrieval**
  - TF-IDF weighting
  - Vector space model
  - Cosine similarity for ranking documents

### Output
- Boolean, phrase, and proximity search results are returned as matching document IDs
- Ranked retrieval returns documents sorted by similarity score
- Results are written to output files in the required coursework format

## Technologies Used
- Python
- XML parsing (`ElementTree`)
- Natural Language Toolkit (Porter Stemmer)
- Custom implementations of indexing and retrieval algorithms
