import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as LangchainPinecone
import os
from functools import lru_cache

# Load environment variables once
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


@lru_cache(maxsize=1)
def initialize_pinecone():
    """
    Initializes Pinecone with the vector database.
    """
    loader = CSVLoader("new_data.csv", encoding='utf-8')
    InData = loader.load_and_split()
    
    pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "steam-games"
    
    if index_name not in pinecone_client.list_indexes().names():
        pinecone_client.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec={"cloud": "aws", "region": "us-east-1"}
        )
        pinecone_index = LangchainPinecone.from_documents(
            InData, 
            OpenAIEmbeddings(), 
            index_name=index_name
        )
    else:
        pinecone_index = LangchainPinecone.from_existing_index(
            index_name, 
            OpenAIEmbeddings()
        )
    
    return pinecone_index

def initialize_chain(pinecone_index):
    """
    Initializes the chain with the retriever, prompt, and model.
    """
    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
    template = """
    You are a knowledgeable assistant for answering gaming-related questions.
    Use the conversation history and retrieved context to answer the current question.
    Ensure your response is refined and relevant to the ongoing discussion.

    Conversation History:
    {history}

    Retrieved Context:
    {context}

    Current Question:
    {question}
    """
    prompt = PromptTemplate.from_template(template)
    retriever = pinecone_index.as_retriever()
    
    return {
        "retriever": retriever,
        "model": model,
        "prompt": prompt,
    }

def refine_query(history, question):
    """
    Refines the user's question based on the context provided in the conversation history.
    Resolves ambiguous references and uses prior responses to clarify queries.
    """
    if "first game" in question.lower() or "second game" in question.lower():
        # Extract recent assistant responses from history
        history_lines = history.split("\n")
        recent_response = next(
            (line.replace("Assistant:", "").strip() for line in reversed(history_lines) if "Assistant:" in line),
            ""
        )
        # Refine question with recent context
        refined_question = f"{question}. Based on the previous response: {recent_response}"
    else:
        # Use the question directly if unrelated
        refined_question = question.strip()

    return refined_question

def get_response(chain, history, question):
    """
    Dynamically generates a response using the refined question and history.
    Handles both follow-up and unrelated queries.
    """
    retriever = chain["retriever"]
    model = chain["model"]
    prompt = chain["prompt"]

    # Refine the question
    refined_question = refine_query(history, question)

    # Retrieve relevant documents from the database
    retrieved_docs = retriever.invoke(refined_question)

    # Construct retrieved context
    if not retrieved_docs:
        retrieved_context = "No relevant results found in the database for this query."
    else:
        retrieved_context = "\n".join(doc.page_content for doc in retrieved_docs)

    # Format the prompt
    formatted_prompt = prompt.format(
        history=history, context=retrieved_context, question=refined_question
    )

    # Generate and return the assistant's response
    response = model.invoke([HumanMessage(content=formatted_prompt)])
    return response.content

def main():
    """
    Main function for the Streamlit application.
    """
    st.title("Steam Games Chatbot")
    st.write("Ask me about Steam games or anything else!")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Load data and initialize components
    pinecone_index = initialize_pinecone()
    chain = initialize_chain(pinecone_index)

    # Build conversation history
    history = "\n".join(
        f"User: {msg['user']}" if 'user' in msg else f"Assistant: {msg['assistant']}"
        for msg in st.session_state.messages
    )

    # Display chat history
    for message in st.session_state.messages:
        if 'user' in message:
            with st.chat_message("user"):
                st.write(message['user'])
        elif 'assistant' in message:
            with st.chat_message("assistant"):
                st.write(message['assistant'])

    # Chat input
    if question := st.chat_input("Ask your question here"):
        # Add user's question to chat
        with st.chat_message("user"):
            st.write(question)
        st.session_state.messages.append({"user": question})

        # Get assistant's response
        response = get_response(chain, history, question)

        # Display assistant's response
        with st.chat_message("assistant"):
            st.write(response)
        st.session_state.messages.append({"assistant": response})

if __name__ == "__main__":
    main()
