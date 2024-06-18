import streamlit as st

import os, glob

# document loaders
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    PyPDFLoader,
)

# text_splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter


# Import openai as main LLM service
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Contextual_compression
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_transformers import (EmbeddingsRedundantFilter,
                                                       LongContextReorder)

from langchain_community.vectorstores import Chroma

# langchain prompts, memory, chains...
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory



TMP_DIR = 'documents'
LOCAL_VECTOR_STORE_DIR = "vectorstore"















def setting_model_params(llm_provider="OpenAI",
                         text_input_API_key="OpenAI API Key - [Get an API key](https://platform.openai.com/account/api-keys)"):

    # API key
    st.session_state.openai_api_key = st.text_input(
        text_input_API_key,
        type="password",
        placeholder="insert your API key",
    )

    st.session_state.LLM_provider = llm_provider

    with st.expander("**Models and parameters**"):
        st.session_state.selected_model = st.selectbox(
            f"Choose {llm_provider} model", ["gpt-3.5-turbo"]
        )

        # model parameters
        st.session_state.temperature = st.slider(
            "temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
        )
        st.session_state.top_p = st.slider(
            "top_p",
            min_value=0.0,
            max_value=1.0,
            value=0.95,
            step=0.05,
        )

def delete_temp_files():
    """delete files from the './data/tmp' folder"""
    files = glob.glob(TMP_DIR + "/*")
    for f in files:
        try:
            os.remove(f)
        except:
            pass


def sidebar_and_document():
    """Create the sidebar and the tabbed pane: the first tab contains a document chooser (create a new vectorstore);
        the second contains a vectorstore chooser (open an old vectorstore)."""
    with st.sidebar:
        st.caption("🚀 A Retrieval Augmented Generation"
                   "chatbot powered by 🔗 Langchain and OpenAI")
        st.write("")

        st.subheader("Retrievers")
        list_retriever_types = ["Contextual compression"]

        st.session_state.retriever_type = st.selectbox(
            f"Select retriever type", list_retriever_types)

        st.divider()

        setting_model_params()

    tab_new_vectorstore, tab_open_vectorstore = st.tabs(["Create a new Vectorstore",
                                                         "Open a saved Vectorstore"])
    with tab_new_vectorstore:
        # 1. Select documnets
        st.session_state.uploaded_file_list = st.file_uploader(
            label="**Select documents**",
            accept_multiple_files=True,
            type=(["pdf", "txt"]),
        )

        # 2. Process documents
        st.session_state.vector_store_name = st.text_input(
            label="**Documents will be loaded, embedded and ingested into a vectorstore (Chroma dB). Please provide a valid dB name.**",
            placeholder="Vectorstore name",
        )
        # 3. Add a button to process documnets and create a Chroma vectorstore

        st.button("Create Vectorstore", on_click=chain_RAG_blocks)  #
        try:
            if st.session_state.error_message != "":
                st.warning(st.session_state.error_message)
        except:
            pass

    with tab_open_vectorstore:
        # Open a saved Vectorstore
        # https://github.com/streamlit/streamlit/issues/1019
        st.write("Please select a Vectorstore:")
        import tkinter as tk
        from tkinter import filedialog

        clicked = st.button("Vectorstore chooser")
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", 1)  # Make dialog appear on top of other windows

        st.session_state.selected_vectorstore_name = ""

        if clicked:
            # Check inputs
            error_messages = []
            if not st.session_state.openai_api_key:
                error_messages.append(
                    f"insert your {st.session_state.LLM_provider} API key"
                )
            if len(error_messages) == 1:
                st.session_state.error_message = "Please " + error_messages[0] + "."
                st.warning(st.session_state.error_message)
            elif len(error_messages) > 1:
                st.session_state.error_message = (
                        "Please "
                        + ", ".join(error_messages[:-1])
                        + ", and "
                        + error_messages[-1]
                        + "."
                )
                st.warning(st.session_state.error_message)

            # if API keys are inserted, start loading Chroma index, then create retriever and ConversationalRetrievalChain
            else:
                selected_vectorstore_path = filedialog.askdirectory(master=root)

                if selected_vectorstore_path == "":
                    st.info("Please select a valid path.")

                else:
                    with st.spinner("Loading vectorstore..."):
                        st.session_state.selected_vectorstore_name = (
                            selected_vectorstore_path.split("/")[-1]
                        )
                        try:
                            # 1. load Chroma vectorestore
                            embeddings = select_embeddings_model()
                            st.session_state.vector_store = Chroma(
                                embedding_function=embeddings,
                                persist_directory=selected_vectorstore_path,
                            )

                            # 2. create retriever
                            st.session_state.retriever = create_compression_retriever(
                                vector_store=st.session_state.vector_store,
                                embeddings=embeddings,
                                base_retriever_search_type="similarity",
                            )

                            # 3. create memory and ConversationalRetrievalChain
                            (
                                st.session_state.chain,
                                st.session_state.memory,
                            ) = create_ConversationalRetrievalChain(
                                retriever=st.session_state.retriever,
                                chain_type="stuff")

                            # 4. clear chat_history
                            clear_chat_history()

                            st.info(
                                f"**{st.session_state.selected_vectorstore_name}** is loaded successfully."
                            )

                        except Exception as e:
                            st.error(e)


####################################################################
#        Process documents and create vector store (Chroma dB)
####################################################################


def langchain_document_loader(TMP_DIR='documents'):
    """
    Load documents from the temporary directory (TMP_DIR).
    Files can be in txt, pdf, CSV or docx format.
    """
    documents = []

    txt_loader = DirectoryLoader(
        TMP_DIR, glob="**/*.txt", loader_cls=TextLoader, show_progress=True
    )
    documents.extend(txt_loader.load())

    pdf_loader = DirectoryLoader(
        TMP_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True
    )
    documents.extend(pdf_loader.load())

    return documents


def split_to_chunks(documents):
    """
    Split documents to chunks using RecursiveCharacterTextSplitter
    For vector store creation
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                   chunk_overlap=100,
                                                   separators=["\n\n", "\n", " ", "", ". ", "! ", "? "],
                                                   add_start_index=True,  # Flag to add start index to each chunk
                                                   )
    chunks = text_splitter.split_documents(documents)
    return chunks


def select_embeddings_model():
    """Select embeddings models: OpenAIEmbeddings or GoogleGenerativeAIEmbeddings."""
    if st.session_state.LLM_provider == "OpenAI":
        embeddings = OpenAIEmbeddings(model='text-embedding-ada-002',
                                      api_key=st.session_state.openai_api_key)

    return embeddings


def vectorstore_backed_retriever(vectorstore,
                                 search_type="similarity",
                                 k=4,
                                 score_threshold=None):
    """
    create a vectorsore-backed retriever
    Parameters:
        search_type: Defines the type of search that the Retriever should perform.
            Can be "similarity" (default), "mmr", or "similarity_score_threshold"
        k: number of documents to return (Default: 4)
        score_threshold: Minimum relevance threshold for similarity_score_threshold (default=None)
    """
    search_kwargs = {}
    if k is not None:
        search_kwargs["k"] = k
    if score_threshold is not None:
        search_kwargs["score_threshold"] = score_threshold

    retriever = vectorstore.as_retriever(
        search_type=search_type, search_kwargs=search_kwargs
    )
    return retriever


def create_compression_retriever(vector_store,
                                 embeddings,
                                 chunk_size=500,
                                 similarity_threshold=None,
                                 base_retriever_search_type="similarity"):
    """Build a ContextualCompressionRetriever.
    We wrap the base_retriever (a Vectorstore-backed retriever) in a ContextualCompressionRetriever.
    The compressor here is a Document Compressor Pipeline, which splits documents
    to smaller chunks, removes redundant documents, filters the top relevant documents,
    and reorder the documents so that the most relevant are at beginning / end of the list.

    Parameters:
        chunk_size (int): Docs will be splitted into smaller chunks using a CharacterTextSplitter with a default chunk_size of 500.
        vector_store: Chroma vector database.
        embeddings: OpenAIEmbeddings or GoogleGenerativeAIEmbeddings.
        base_retriever_search_type: search_type in ["similarity", "mmr", "similarity_score_threshold"], default = similarity.
        similarity_threshold : similarity_threshold of the  EmbeddingsFilter. default =None
    """
    base_retriever = vectorstore_backed_retriever(
        vectorstore=vector_store,
        search_type=base_retriever_search_type,
        k=12,
        score_threshold=None,
    )

    # 1. splitting docs into smaller chunks
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=0, separator=". "
    )

    # 2. removing redundant documents
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)

    # 3. filtering based on relevance to the query
    relevant_filter = EmbeddingsFilter(
        embeddings=embeddings, k=16, similarity_threshold=similarity_threshold
    )

    # 4. Reorder the documents
    # Less relevant document will be at the middle of the list and more relevant elements at beginning / end.
    reordering = LongContextReorder()

    # 5. create compressor pipeline and retriever
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, redundant_filter, relevant_filter, reordering]
    )
    retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=base_retriever
    )

    return retriever


def chain_RAG_blocks():
    with st.spinner("Creating vectorstore..."):
        # Check inputs
        error_messages = []
        if not st.session_state.openai_api_key:
            error_messages.append(
                f"insert your {st.session_state.LLM_provider} API key")

        if not st.session_state.uploaded_file_list:
            error_messages.append("select documents to upload")

        if st.session_state.vector_store_name == "":
            error_messages.append("provide a Vectorstore name")

        if len(error_messages) == 1:
            st.session_state.error_message = "Please " + error_messages[0] + "."
        elif len(error_messages) > 1:
            st.session_state.error_message = (
                    "Please "
                    + ", ".join(error_messages[:-1])
                    + ", and "
                    + error_messages[-1]
                    + "."
            )
        else:
            st.session_state.error_message = ""
            try:
                # 1. Delete old temp files
                delete_temp_files()

                # 2. Upload selected documents to temp directory
                if st.session_state.uploaded_file_list is not None:
                    for uploaded_file in st.session_state.uploaded_file_list:
                        error_message = ""
                        try:
                            temp_file_path = os.path.join(
                                TMP_DIR, uploaded_file.name
                            )
                            with open(temp_file_path, "wb") as temp_file:
                                temp_file.write(uploaded_file.read())
                        except Exception as e:
                            error_message += f'{e}'
                    if error_message != "":
                        st.warning(f"Errors: {error_message}")
                    # 3. Load documents with Langchain loaders
                    documents = langchain_document_loader()

                    # 4. Split documents to chunks
                    chunks = split_to_chunks(documents)
                    # 5. Embeddings
                    embeddings = select_embeddings_model()
                    # 6. Create a vectorstore
                    persist_directory = LOCAL_VECTOR_STORE_DIR + "/" + st.session_state.vector_store_name

                    try:
                        st.session_state.vector_store = Chroma.from_documents(
                            documents=chunks,
                            embedding=embeddings,
                            persist_directory=persist_directory)

                        st.info(
                            f"Vectorstore **{st.session_state.vector_store_name}** is created succussfully.")

                        # 7. Create retriever
                        st.session_state.retriever = create_compression_retriever(vector_store=st.session_state.vector_store,
                                                                                  embeddings=embeddings,
                                                                                  base_retriever_search_type="similarity",)
                        # 3. create memory and ConversationalRetrievalChain
                        st.session_state.chain,st.session_state.memory = create_ConversationalRetrievalChain(
                            retriever=st.session_state.retriever,
                            chain_type="stuff")

                        # 4. clear chat_history
                        clear_chat_history()

                        st.info(
                            f"**{st.session_state.selected_vectorstore_name}** is loaded successfully."
                        )

                    except Exception as e:
                        st.error(e)

            except Exception as error:
                st.error(f"An error occurred: {error}")


####################################################################
#                       Create memory
####################################################################


def create_memory(model_name="gpt-3.5-turbo", memory_max_token=None):
    """Creates a ConversationSummaryBufferMemory for gpt-3.5-turbo
    Creates a ConversationBufferMemory for the other models"""

    if model_name == "gpt-3.5-turbo":
        if memory_max_token is None:
            memory_max_token = 1024  # max_tokens for 'gpt-3.5-turbo' = 4096
        memory = ConversationSummaryBufferMemory(
            max_token_limit=memory_max_token,
            llm=ChatOpenAI(
                model_name="gpt-3.5-turbo",
                openai_api_key=st.session_state.openai_api_key,
                temperature=0.1,
            ),
            return_messages=True,
            memory_key="chat_history",
            output_key="answer",
            input_key="question",
        )
    else:
        memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history",
            output_key="answer",
            input_key="question",
        )
    return memory


####################################################################
#          Create ConversationalRetrievalChain with memory
####################################################################


def answer_template(language="english"):
    """Pass the standalone question along with the chat history and context
    to the `LLM` wihch will answer."""

    template = f"""Answer the question at the end, using only the following context (delimited by <context></context>).
Your answer must be in the language at the end. 

<context>
{{chat_history}}

{{context}} 
</context>

Question: {{question}}

Language: {language}.
"""
    return template


def create_ConversationalRetrievalChain(retriever,chain_type="stuff"):
    """Create a ConversationalRetrievalChain.
    First, it passes the follow-up question along with the chat history to an LLM which rephrases
    the question and generates a standalone query.
    This query is then sent to the retriever, which fetches relevant documents (context)
    and passes them along with the standalone question and chat history to an LLM to answer.
    """

    # 1. Define the standalone_question prompt.
    # Pass the follow-up question along with the chat history to the `condense_question_llm`
    # which rephrases the question and generates a standalone question.

    condense_question_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template="""Given the following conversation and a follow up question, 
rephrase the follow up question to be a standalone question, in its original language.\n\n
Chat History:\n{chat_history}\n
Follow Up Input: {question}\n
Standalone question:""",
    )

    # 2. Define the answer_prompt
    # Pass the standalone question + the chat history + the context (retrieved documents)
    # to the `LLM` wihch will answer

    answer_prompt = ChatPromptTemplate.from_template(answer_template())

    # 3. Add ConversationSummaryBufferMemory for gpt-3.5, and ConversationBufferMemory for the other models
    memory = create_memory(st.session_state.selected_model)

    # 4. Instantiate LLMs: standalone_query_generation_llm & response_generation_llm
    standalone_query_generation_llm = ChatOpenAI(
            api_key=st.session_state.openai_api_key,
            model=st.session_state.selected_model,
            temperature=0.1,
        )
    response_generation_llm = ChatOpenAI(
            api_key=st.session_state.openai_api_key,
            model=st.session_state.selected_model,
            temperature=st.session_state.temperature,
            model_kwargs={"top_p": st.session_state.top_p},
        )
    # 5. Create the ConversationalRetrievalChain

    chain = ConversationalRetrievalChain.from_llm(
        condense_question_prompt=condense_question_prompt,
        combine_docs_chain_kwargs={"prompt": answer_prompt},
        condense_question_llm=standalone_query_generation_llm,
        llm=response_generation_llm,
        memory=memory,
        retriever=retriever,
        chain_type=chain_type,
        verbose=False,
        return_source_documents=True,
    )

    return chain, memory


def clear_chat_history():
    """clear chat history and memory."""
    # 1. re-initialize messages
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": 'How can I assist you today?',
        }
    ]
    # 2. Clear memory (history)
    try:
        st.session_state.memory.clear()
    except:
        pass


def get_response_from_LLM(prompt):
    """invoke the LLM, get response, and display results (answer and source documents)."""
    try:
        # 1. Invoke LLM
        response = st.session_state.chain.invoke({"question": prompt})
        answer = response["answer"]

        # 2. Display results
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            # 2.1. Display anwser:
            st.markdown(answer)

            # 2.2. Display source documents:
            with st.expander("**Source documents**"):
                documents_content = ""
                for document in response["source_documents"]:
                    try:
                        page = " (Page: " + str(document.metadata["page"]) + ")"
                    except:
                        page = ""
                    documents_content += (
                        "**Source: "
                        + str(document.metadata["source"])
                        + page
                        + "**\n\n"
                    )
                    documents_content += document.page_content + "\n\n\n"

                st.markdown(documents_content)

    except Exception as e:
        st.warning(e)


####################################################################
#                         Chatbot
####################################################################
def chatbot():
    st.set_page_config(page_title="Chat With Your Data", layout="wide")

    # Sidebar

    # Main content
    st.title("RAG-Powered App with OpenAI")

    sidebar_and_document()
    st.divider()
    col1, col2 = st.columns([7, 3])
    with col1:
        st.subheader("Chat with your data")
    with col2:
        st.button("Clear Chat History", on_click=clear_chat_history)

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": 'How can I assist you today?',
            }
        ]
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        if (
            not st.session_state.openai_api_key
            and not st.session_state.google_api_key
            and not st.session_state.hf_api_key
        ):
            st.info(
                f"Please insert your {st.session_state.LLM_provider} API key to continue."
            )
            st.stop()
        with st.spinner("Running..."):
            get_response_from_LLM(prompt=prompt)


if __name__ == "__main__":
    chatbot()

