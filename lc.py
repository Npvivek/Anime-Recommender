# Cell 1
import os
os.environ['WANDB_MODE'] = 'disabled'
import warnings
warnings.filterwarnings('ignore')

!pip install --upgrade transformers sentence-transformers huggingface-hub
!pip install langchain langchain-huggingface
!pip install chromadb tiktoken
!pip install langchain-community
!pip install datasets

# Cell 2
import pandas as pd
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate

# Print library versions to confirm
import transformers
print(f"Transformers version: {transformers.__version__}")
import sentence_transformers
print(f"SentenceTransformers version: {sentence_transformers.__version__}")
import huggingface_hub
print(f"HuggingFace Hub version: {huggingface_hub.__version__}")
import langchain
print(f"LangChain version: {langchain.__version__}")

# Cell 3
anime = pd. read_csv('anime_with_synopsis.csv')
anime.head()

# Cell 4
if 'sypnopsis' in anime.columns:
    anime.rename(columns={'sypnopsis': 'synopsis'}, inplace=True)


anime = anime.dropna(subset=['Name', 'synopsis', 'Genres'])


anime = anime[~anime['synopsis'].str.contains("No synopsis information", na=False)]

# Cell 5
# Combine the information
anime['combined_info'] = anime.apply(
    lambda row: f"Title: {row['Name']}. Overview: {row['synopsis']} Genres: {row['Genres']}",
    axis=1
)
anime['combined_info'][0]

# Cell 6
#Save processed dataset - combined_info for Langchain
anime[['combined_info']].to_csv('anime_updated.csv', index=False)

# Cell 7
processed_anime = pd.read_csv('anime_updated.csv')
processed_anime.head()

# Cell 8
from sentence_transformers import InputExample

# Load a pre-trained SentenceTransformer model
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
embedding_model = SentenceTransformer(model_name)

# Prepare the data for unsupervised SimCSE fine-tuning
train_sentences = anime['combined_info'].tolist()

# Create InputExample instances with two identical sentences
train_examples = [InputExample(texts=[sent, sent]) for sent in train_sentences]

# Create a DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Use the MultipleNegativesRankingLoss
train_loss = losses.MultipleNegativesRankingLoss(embedding_model)

# Fine-tune the model using SimCSE approach
embedding_model.train()  # Activate training mode to enable dropout
embedding_model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    show_progress_bar=True
)

# Save the fine-tuned model to a directory
embedding_model.save('fine_tuned_model')

# Cell 9
# Use the fine-tuned model by specifying the path
embeddings = HuggingFaceEmbeddings(model_name='fine_tuned_model')

# Load the data using LangChain's CSVLoader
loader = CSVLoader(file_path="anime_updated.csv")
data = loader.load()

# Split the documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(data)

# Create a vector store using Chroma
docsearch = Chroma.from_documents(texts, embeddings)


# Cell 10
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.llms import HuggingFacePipeline

# Load the Flan-T5 model and tokenizer
model_name = 'google/flan-t5-large'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Create a pipeline for text generation
hf_pipeline = pipeline(
    'text2text-generation',
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    temperature=0.9,  # Increased temperature for more creativity
    top_p=0.95,       # Increased top_p to allow more diverse tokens
    repetition_penalty=1.1,
    do_sample=True
)

# Wrap the pipeline in a LangChain LLM
local_llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Cell 11
# Define the prompt template
template = """You are an anime recommender system that helps users find anime that match their preferences.
Use the following context to answer the question at the end.
For each recommendation, suggest three anime films with a short description of the plot and why the user might like them.
If you don't know the answer, say that you don't know; don't try to make up an answer.

{context}

Question: {question}
Your response:"""

PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

# Define the retriever with increased k
retriever = docsearch.as_retriever(search_kwargs={"k": 10})

# Set up the RetrievalQA chain with the local LLM
qa = RetrievalQA.from_chain_type(
    llm=local_llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT},
    verbose=True  # Enable verbose logging
)

# Cell 12
query = "I'm looking for a dark fantasy anime where man eating titans are involved . What could you suggest to me?"
result = qa.invoke({"query": query})  # Updated method call
print(result['result'])  # Print the recommendations

# Cell 13
# Print the retrieved documents
print("Retrieved Documents:")
for doc in result['source_documents']:
    print(doc.page_content)
    print("\n---\n")

# Cell 14
result['source_documents'][0]

# Cell 15
# Manually construct the prompt
sample_context = " ".join([doc.page_content for doc in result['source_documents']])
test_prompt = f"""You are an anime recommender system that helps users find anime that match their preferences.
Use the following context to answer the question at the end.
For each recommendation, suggest three anime films with a short description of the plot and why the user might like them.
If you don't know the answer, say that you don't know; don't try to make up an answer.

{sample_context}

Question: {query}
Your response:"""

# Generate a response using the local LLM
response = local_llm(test_prompt)
print(response)

# Cell 16
# Define user information
age = 23
gender = 'male'

# Update the prompt to include user info
template_prefix = f"""You are an anime recommender system that helps users find anime that match their preferences.
Use the following context and the user's information to answer the question at the end.
If you don't know the answer, say that you don't know; don't try to make up an answer.

User Information:
- Age: {age}
- Gender: {gender}

{{context}}"""

template_suffix = """
Question: {question}
Your response:"""

COMBINED_PROMPT = template_prefix + template_suffix

PROMPT = PromptTemplate(template=COMBINED_PROMPT, input_variables=["context", "question"])

# Define the retriever with increased k
retriever = docsearch.as_retriever(search_kwargs={"k": 10})

# Update the RetrievalQA chain with the local LLM
qa = RetrievalQA.from_chain_type(
    llm=local_llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT},
    verbose=True  # Optionally enable verbose logging
)

# Get the personalized recommendations
result = qa.invoke({'query': query})

# Print the recommendations
print(result['result'])

# Cell 17
result['source_documents']

# Cell 18
queries = [
    "I'm looking for a romantic comedy anime. Any suggestions?",
    "Can you recommend an anime with strong female leads?",
    "What are some good sci-fi anime with space battles?",
    "I'm interested in anime that explore psychological themes.",
    "Suggest some anime movies with pirates."
]


# Cell 19
for query in queries:
    print(f"Query: {query}")
    result = qa.invoke({"query": query})
    print("Recommendations:")
    print(result['result'])
    print("\n" + "="*50 + "\n")


# Cell 20
import nbformat

def extract_code_from_ipynb(ipynb_file, output_file):
    with open(ipynb_file, 'r', encoding='utf-8') as file:
        notebook = nbformat.read(file, as_version=4)
        
    code_cells = [cell['source'] for cell in notebook['cells'] if cell['cell_type'] == 'code']
    
    with open(output_file, 'w', encoding='utf-8') as file:
            for i, code in enumerate(code_cells, 1):               
                file.write(f"# Cell {i}\n")
                file.write(code)
                file.write('\n\n')


# Replace 'notebook.ipynb' and 'output.py' with your file names
extract_code_from_ipynb('lc.ipynb', 'lc.py')

# Cell 21


