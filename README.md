# AI_Telecom_Challenge

## Inference

![arch_v3](https://github.com/user-attachments/assets/da3ef0ce-c1b9-4d53-a1ea-b798f60f70de)


### Hybrid Retriever
I defined a custom hybrid_retriever class that inherits from BaseRetriever. This retriever performs a hybrid search by combining both semantic search (vector-based) and keyword search (keyword-based). It allows you to control the behavior of the retrieval using either an "AND" or "OR" mode, which determines how the results from the vector and keyword retrievers are combined.

Constructor (__init__):
Takes in two retrievers: vector_retriever (semantic/vector-based search) and keyword_retriever (keyword-based search).
mode: Determines how results are combined. It can either be "AND" (retrieve nodes common to both vector and keyword search) or "OR" (retrieve nodes from either search).
Raises a ValueError if the mode is neither "AND" nor "OR".
_retrieve:
The _retrieve method accepts a QueryBundle and returns a list of nodes (List[NodeWithScore]) that match the query.

Retrieve from Vector and Keyword Retrievers:

It first retrieves results from the vector-based retriever (self._vector_retriever.retrieve) and the keyword-based retriever (self._keyword_retriever.retrieve).
Combine the Results:

A set of node IDs is extracted from the vector-based results (vector_ids) and keyword-based results (keyword_ids).
The combined_dict stores all retrieved nodes, indexed by their node IDs, allowing easy look-up.
Intersection/Union Based on Mode:

In "AND" mode: Only nodes that exist in both vector_ids and keyword_ids are kept.
In "OR" mode: Nodes from either vector_ids or keyword_ids are kept.
Return Nodes:

The final list of nodes (retrieve_nodes) is created from the combined results and returned.

#### BM25 Retriever

Create the BM25 retriever:

BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=2) initializes a BM25 retriever, setting the top-2 most similar documents to be retrieved based on BM25 similarity.
Persist the BM25 retriever:

bm25_retriever.persist("./bm25_retriever") saves the BM25 retriever object to disk at the specified path ./bm25_retriever for later use.


## Data Generation for SFT using RAG (Retrieval Augmented Generation)

I used ChromaDB as Vector Database to store the documents. he PersistentClient in ChromaDB is being used to create a connection to the database system that manages vector data (i.e., numerical representations of documents or text). ChromaDB is a specialized database optimized for handling vectors, which are often used in machine learning and search applications.
The chroma_collection=chroma_collection is passing the ChromaDB collection (which was either created or retrieved earlier) into the ChromaVectorStore. This collection is where all the vectorized representations of your documents will be stored.
The ChromaVectorStore is essentially a wrapper that provides an interface to work with the underlying ChromaDB collection. It allows you to:
Add vectorized data to the collection (e.g., the document vectors created from text).
Search or query the collection using vector similarity (e.g., finding documents that are similar to a given query based on their vector representation).
Manage the stored vectors (e.g., deleting or updating them).
A StorageContext is an object that acts as a management layer for the storage of data, typically vectors in your case. It organizes how the vectors are stored, retrieved, and used within the application.
It abstracts the details of where and how data is stored, making it easier for your application to interact with storage mechanisms without dealing with low-level details like database queries, file handling, etc.
A VectorStoreIndex is a data structure that stores and organizes vectorized documents for efficient retrieval. It allows you to search for documents or data by comparing vector representations (e.g., similarity-based search).
This is particularly useful in tasks like information retrieval, question answering, or semantic search, where you want to find documents based on their vector similarity to a query.

I used reranking to further improve the results of retrieval. This model reranks top 15 chunks with top chunk as most relevant document related to query. SentenceTransformerRerank is a re-ranking class or function that uses a transformer-based model to re-rank a set of results based on their relevance to a query.
It applies a cross-encoder model, which performs pairwise comparisons between the query and each document. Instead of encoding the query and documents separately, a cross-encoder jointly processes both and computes a similarity score for each document based on its relevance to the query.


I pass the question through retriever and attach the context given by retriver to question. This is then passed to custom prompt generation template.  


### Parent Document Retriever

I also implement Parent Document Retriever using Recursive retrieval and Node references in llamaindex. This technique search relevant documents with small chunks and return the original text to the chunk. This text is used a context. The code can be found in Parent_document_retriever.ipynb file. 


## Prompt Generation

I used a Chain of Thought (COT) prompt that includes six crucial steps. These steps direct the Large Language Model (LLM) to focus on key details and then reason by taking those details into account. It looks like:
Instructions:
    Step 1. Read the context carefully to identify the most relevant information.
    Step 2. Pay special attention to key terms in the question, such as "main," "primary," "most important," etc. These terms indicate that the answer should focus on the most 
            significant aspect mentioned in the context.
    Step 3. Focus on keywords and phrases in the context that directly relate to the question being asked.
    Step 4. Match the context to the options provided and choose the option that is most explicitly supported by the context and aligns with the key terms in the question.
    Step 5. If the context does not provide clear support for any option, choose the option that logically fits the context and the question.
    Step 6. Pay attention to abbreviations related to wireless communications and 3GPP standards'

I include four options in the prompt, with a fifth option provided if applicable. Additionally, I incorporate context in the prompt that is highly relevant to both the question and the four options. These contexts are generated using Retrieval Augmented Generation (RAG), as explained in the previous section.


## FineTuning Techniques

I used 4-bit quantization using the BitsAndBytesConfig class from the BitsAndBytes library (often used for efficient model quantization). This configuration is tailored for low-precision model inference to reduce memory usage and computation cost while maintaining performance. Let's break down the components of this configuration:

Key Components:
load_in_4bit=True:

This option tells the system to load the model with 4-bit quantization. This reduces the memory requirements of the model, as the weights will be stored in 4-bit precision rather than the standard 32-bit or 16-bit precision.
4-bit quantization is used to significantly reduce the model size and memory footprint, which is particularly useful for running large models on hardware with limited resources (e.g., GPUs with lower memory capacity).
bnb_4bit_quant_type='nf4':

This specifies the type of 4-bit quantization. The 'nf4' refers to a specific quantization scheme called "Normalized Floating Point 4-bit" (NF4).
NF4 is a method of quantizing weights where the model weights are represented in a 4-bit format with a floating-point style encoding. This scheme allows for efficient storage while preserving model performance by maintaining a good balance between precision and size.
bnb_4bit_compute_dtype='float16':

This specifies that the computations will be performed using 16-bit floating-point precision (float16), which is commonly used in machine learning to reduce memory usage and speed up training or inference.
This configuration suggests that even though the model weights are quantized to 4-bit, the actual computations during inference or training will be done using 16-bit precision, which helps maintain numerical stability and performance.
bnb_4bit_use_double_quant=True:

This option enables double quantization for 4-bit quantization. Double quantization is a technique that applies an additional layer of quantization to further reduce the model size while mitigating the loss of accuracy that might come from aggressive quantization.
It helps achieve further compression of the model without significantly compromising performance, which is crucial when working with large models in resource-constrained environments.

I used gradient checkpointing, which is a memory optimization technique that trades compute for memory. It allows you to reduce the amount of memory required to store intermediate activations during the forward pass of training, by saving only a subset of these activations (the "checkpoints").
prepare_model_for_kbit_training is a function (likely specific to a library or framework you're using) that prepares the model for low-bit quantization (k-bit training). By setting use_gradient_checkpointing=True, this ensures that gradient checkpointing is applied while training the model with reduced precision
I used LoRA to fine-tune pre-trained large models, which efficiently by injecting trainable low-rank matrices into specific parts of the model, instead of fine-tuning all parameters.
This creates a PEFT (Parameter-Efficient Fine-Tuning) model by applying the LoRA configuration to the base model. The get_peft_model function integrates the low-rank adaptation with the model to make the fine-tuning more efficient by only training a small set of parameters (those related to the low-rank matrices).

A data collator is responsible for batching and preparing the data for training. Here:
tokenizer=tokenizer: The tokenizer used to process input sequences.
mlm=False: Since this is a causal language modeling task, mlm (Masked Language Modeling) is set to False because causal LM does not use masked tokens

