# Quora Duplicate Question Scanner(NLP based Project)

> A production-ready Semantic Duplicate Question Detection & Live Quora Search System powered by SBERT, FAISS, FastAPI, and React.

This project intelligently detects duplicate questions by understanding their **meaning**, not just matching keywords. It combines **Semantic Search** over a local database of **533,361+ indexed Quora questions** with a **Live Quora Search**, allowing users to instantly check whether a similar question already exists both in the local dataset and on Quora.

---

# The Real Problem

Millions of users ask questions every day on platforms like **Quora**, **Stack Overflow**, **Reddit**, and customer support portals.

Many of these questions are duplicates but are written differently.

### Example

```
How can I learn Python?
```

```
What is the best way to start learning Python?
```

```
How do I begin learning Python?
```

Although these questions use different words, they ask exactly the same thing.

Traditional keyword-based search often fails because it searches for matching words instead of understanding the actual meaning.

This leads to:

- Duplicate questions
- Repeated answers
- Poor search experience
- Knowledge fragmentation
- Increased moderation effort
- Wasted storage and indexing resources

This project solves this problem using Semantic Search powered by modern NLP.

---

# Solution

The system converts every question into a **768-dimensional semantic vector** using **Sentence-BERT (all-mpnet-base-v2)**.

Whenever a user enters a new question, the application simultaneously performs two searches:

### Local Semantic Search

Searches over **533,361 indexed Quora questions** using **FAISS Approximate Nearest Neighbor Search** to retrieve the most semantically similar questions.

### Live Quora Search

Performs a real-time search on **Quora.com** to verify whether the question (or a similar one) already exists publicly on Quora.

Users can compare both results side-by-side.

This provides a much richer experience than traditional duplicate detection systems.

---

# How It Works

```
                 User Question
                       │
                       ▼
         Sentence-BERT Embedding (768-D)
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
      FAISS Local Search     Live Quora Search
             │                   │
             ▼                   ▼
     Top-K Similar Questions   Existing Questions
             │                   │
             └─────────┬─────────┘
                       ▼
             Display Results Together
```

---

# Example

### User Query

```
How can I learn Python?
```

### Local Semantic Results

```
1. How do I learn Python?
Similarity : 97.84%

2. I want to learn Python where do I start?
Similarity : 92.64%

3. How can I learn Python on my own?
Similarity : 92.10%

4. What is the best way to learn Python?
Similarity : 90.08%

5. How do I learn Python in an easy way?
Similarity : 90.07%
```

### Live Quora Search

The application also searches Quora in real time and displays related questions that already exist on the platform.

---

# Features

- Semantic Duplicate Question Detection
- Sentence-BERT (SBERT) Embeddings
- FAISS Vector Similarity Search
- Live Quora Search Integration
- Check whether a question already exists on Quora
- Real-Time Similarity Scores
- Top-K Similar Question Retrieval
- FastAPI REST API
- React + TypeScript Frontend
- Modern Dark UI
- Millisecond Search Performance
- Supports 533K+ Indexed Questions
- Production-Ready Architecture

---

# Why This Project Is Different

Most duplicate question detection projects only search within a local dataset.

This project goes one step further.

For every user query it:

- Understands the semantic meaning using SBERT.
- Searches over **533K+ indexed questions** using FAISS.
- Retrieves the Top-K most similar questions with similarity scores.
- Simultaneously performs a Live Quora Search.
- Lets users compare local semantic matches with actual questions already available on Quora.

This makes the system useful not only for duplicate detection but also for **content research, knowledge discovery, and question validation**.

---

# Tech Stack

## Machine Learning

- Sentence Transformers (SBERT)
- all-mpnet-base-v2
- FAISS
- NumPy
- Pandas

## Backend

- FastAPI
- Uvicorn
- Pydantic

## Frontend

- React
- TypeScript
- Tailwind CSS

## Tools

- Git
- GitHub
- VS Code
- Google Colab

---

# Project Structure

```
Quora-Duplicate-Questions/
│
├── api/
├── app/
├── data/
│   ├── processed/
│   ├── embeddings/
│   └── raw/
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── build_index.py
│   └── inference.py
│
├── requirements.txt
└── README.md
```

---

# NLP Pipeline

```
Raw Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Text Preprocessing
      │
      ▼
Sentence-BERT Embeddings
      │
      ▼
FAISS Index Building
      │
      ▼
User Question
      │
      ▼
Sentence Embedding
      │
      ▼
Nearest Neighbor Search
      │
      ├──────────────► Live Quora Search
      ▼
Top-K Similar Questions
      │
      ▼
Similarity Scores
      │
      ▼
Results Display
```

---

# Dataset

| Metric | Value |
|---------|-------|
| Dataset | Quora Question Pairs |
| Processed Rows | 404,348 |
| Total Questions | 808,696 |
| Unique Questions | 533,361 |
| Embedding Dimension | 768 |

---

# Real-World Applications

- Quora Duplicate Question Detection
- Stack Overflow Similar Question Search
- Reddit Question Search
- Customer Support Ticket Deduplication
- FAQ Recommendation Systems
- Enterprise Knowledge Base Search
- AI Assistants
- Chatbots
- Internal Company Search Engines
- Community Moderation Tools
- Content Research Platforms
- Live Question Validation Systems

---

# Screenshots

Add application screenshots inside the `assets` folder.

```
assets/home.png
assets/search_results.png
assets/live_quora_search.png
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/aman-pathak1/Quora-Duplicate-Questions.git
```

Move into the project directory

```bash
cd Quora-Duplicate-Questions
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Run

Backend

```bash
uvicorn api.main:app --reload
```

Frontend

```bash
npm install
npm run dev
```

---

# Future Improvements

- Duplicate / Not Duplicate Classification
- Cross-Encoder Re-ranking
- Hybrid BM25 + SBERT Retrieval
- Multilingual Search
- Voice Question Search
- Docker Deployment
- AWS / Azure / GCP Deployment
- Authentication
- Search Analytics Dashboard

---

# Project Highlights

- 533K+ Semantic Search Index
- Sentence-BERT Embeddings
- FAISS Approximate Nearest Neighbor Search
- Live Quora Search
- FastAPI + React Full Stack Application
- Real-Time Similarity Scores
- Semantic Duplicate Question Detection
- Millisecond Response Time
- Production-Ready Codebase

---

# License

This project is licensed under the **MIT License**.

---

# Support

If you found this project useful, consider giving it a star on GitHub.
