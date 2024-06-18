# Segmentation Model with TensorFlow and TFRecords
## Overview
This repository contains code for RAG chatbot in Langchain creation powered by OpenAI API. It enables us to upload documents in txt, pdf formats and chat with your data. Relevant documents 
are be retrieved and sent to the LLM along with any follow-up questions for accurate answers.
<br>Additionally, there is a **user interface** using streamlit application..

## Folder structure
```
my_project/
│
├── documents/
│ ├── document_1.txt
│ ├── document_2.txt
│ └── ...
├── docs/
│ ├── document_1.txt
│ ├── document_2.txt
│ └── ...
├── vectorstore/
│ ├── tests/
│ ├── ...
│ └── ...
├── rag_app.py
├── Dockerfile
└── requirments.txt
```
### Files Description
- `documents/`: a temporary directory with txt files that is erased after we create a new vector store.
- `vectorstore/`: a Directory containing vector stores.
  - `tests`: a vector ssote for my particular example.
- `docs/`: a Directory that is used just as an example for docs uploading.
- `rag_app.py`: Script to load Chatbot and run the streamlit app.
- `Dockerfile`: contains all information for Docker.
- `model.py`: Defines the U-Net model architecture and the class for handling our datasets.


## Requirements
To be albe to use the model yuo should have two folders (they can be empty):
* vectorestore/
* documents/

This project requires Python 3 and the following Python libraries installed: *langchain, langchain-openai, langchain-community, chromadb, streamlit*
<br>All necessary **dependencies** You can install using pip:
```python
pip install -r requirements.txt
```

## Intro
To start using the app and see the users interface you simple need to write:
```python
streamlit run rag_app.py
```

## Overview
The App is pretty self explanatory and you can see how it is going to look like **below**:

![image](https://github.com/PlatMurav/RAG-chatbot/assets/112167233/5e881b0e-4559-4f07-b1b6-f834f2cab43c)
### In the sidebar (*Models and parameters*) you need:
1. Choose an LLM (*we have just GPT-3.5 yet*)
2. Adjust its parameters (*temperature and top_p*)
3. Insert your **API key**.

### Chroma vectorstores
As you've seen above you have two options:
1. Create a new vector store
2. Open saved a vector store

#### Creation
If you choose the first one you'll see the window where you can list any necessary documents in txt and pdf formats. Chat uses those documents to answer you specific questions.
<br>The documents will be uploaded and then you simply need to write a name for the folder for your new vector store.
![image](https://github.com/PlatMurav/RAG-chatbot/assets/112167233/5c497524-8b83-421b-9b7b-f502826df6ab)

#### Loading
If you choose "*open a saved Vector store*" you need to find the "*vectorstore*" folder and choose the vector store you created (folder).
<br>Then you'll see:
<br>![image](https://github.com/PlatMurav/RAG-chatbot/assets/112167233/353ff147-5ce8-431e-a672-2cbe12f0a7d2)

### Chatbot
If you did all previous steps correctly you can start asking questions.
<br>Here is an example below:
![image](https://github.com/PlatMurav/RAG-chatbot/assets/112167233/04ed648a-a527-477a-8430-ff84b55d9e5d)
