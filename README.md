# Case study 1: APRIO

## What is it?

### Data

The data delivered by EFSA consists of
1. `data/0-papers-raw/`: 748 PDF articles,
1. `data/1-papers-processed/`: JSON OCR output for these articles,
1. `data/2-papers-text/`: text contained in the JSON files.

## Preprocessing

1. `cs1_metadata.ipynb`: use `data/1-papers-processed/` to extract metadata such as
      * title
      * authors
      * abstract
      * panel?
      * keywords?
   Results are in `data/3-papers-metadata/` as JSON files.
1. `tf_idf.ipynb`: use `data/2-papers-text/` to compute a TF-IDF representation for
   each document.  Results are in `data/4-papers-tfidf/` as text representation of
   real-valued document vectors.
