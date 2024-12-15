# Anime Recommender System

This project is a machine learning-based anime recommender system designed to help users discover anime titles that match their preferences. By leveraging embeddings, fine-tuning, and retrieval-augmented generation (RAG), the system suggests relevant anime based on user queries. This project runs entirely in a Jupyter Notebook and uses open-source tools like LangChain, Hugging Face, and Sentence Transformers.

## Features

- **Fine-tuned Embeddings**: Custom embeddings are created using a fine-tuned SentenceTransformer model for better similarity matching.
- **Anime Dataset**: Parses and preprocesses a dataset of anime titles, synopses, and genres.
- **Recommendation Engine**: Combines vector search with a large language model (LLM) to generate human-like recommendations.
- **Interactive User Prompts**: Provides personalized anime recommendations based on specific user queries.
- **Extensibility**: Can be easily adapted for other datasets and recommendation use cases.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.7 or higher
- Jupyter Notebook
- Necessary libraries (see below for installation commands)

### Installation

Run the following commands to install the required libraries:

```bash
!pip install --upgrade transformers sentence-transformers huggingface-hub
!pip install langchain langchain-huggingface
!pip install chromadb tiktoken
!pip install langchain-community
!pip install datasets
```

### Running the Notebook

1. Clone this repository and navigate to the notebook file (`anime_recommender.ipynb`).
2. Open the notebook in Jupyter Notebook or Google Colab.
3. Execute all cells sequentially to:
   - Preprocess the dataset
   - Fine-tune the embedding model
   - Create a vector store
   - Query the system with anime-related questions

## Workflow Overview

### Step 1: Preprocessing the Dataset

- The dataset (`anime_with_synopsis.csv`) is cleaned by removing rows with missing values or placeholder text (e.g., "No synopsis information").
- Data is combined into a single field, `combined_info`, containing the title, synopsis, and genres.

### Step 2: Fine-Tuning the Embedding Model

- A SentenceTransformer model (`all-MiniLM-L6-v2`) is fine-tuned using SimCSE with a MultipleNegativesRankingLoss objective.
- This enhances the model’s ability to identify similar anime titles based on semantic similarity.

### Step 3: Building the Vector Store

- Preprocessed data is split into smaller chunks using LangChain’s `CharacterTextSplitter`.
- A Chroma vector store is created for fast similarity search over the dataset.

### Step 4: Querying with an LLM

- The fine-tuned embeddings are paired with the Flan-T5 LLM (`google/flan-t5-large`) to handle user queries.
- LangChain’s `RetrievalQA` pipeline is used to fetch the most relevant documents and generate human-like recommendations.

## Example Queries

Here are some example queries you can try in the notebook:

1. **"I'm looking for a dark fantasy anime where man-eating titans are involved."**
2. **"Can you recommend an anime with strong female leads?"**
3. **"Suggest some good sci-fi anime with space battles."**
4. **"I'm interested in anime that explore psychological themes."**

## Output

- The system provides three anime recommendations for each query, with a brief plot description and reasoning for each suggestion.
- Additionally, the retrieved documents from the dataset are displayed for transparency.

---

**Contributors**: If you have suggestions or would like to contribute, feel free to open an issue or submit a pull request. Happy coding!

