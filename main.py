# This is a sample Python script.
import os

os.environ['OPENAI_API_KEY'] = 'PASTE YOUR API KEY HERE'
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext, PromptHelper
from langchain_openai import ChatOpenAI
from llama_index.core import Settings
import gradio as gr
import sys


def init_index(directory):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 2
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs,
                                 chunk_overlap_ratio=0.1, chunk_size_limit=chunk_size_limit)
    Settings.llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs)
    documents = SimpleDirectoryReader(directory).load_data()
    service_context = ServiceContext.from_defaults(llm=Settings.llm, prompt_helper=prompt_helper)
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    index.storage_context.persist("data")

    return index


def chatbot(text):
    index = StorageContext.from_defaults(persist_dir="data")
    load_index = load_index_from_storage(index)
    index_query = load_index.as_query_engine()
    response = index_query.query(text)

    return response.response


if __name__ == "__main__":
    init_index("docs")

    iface = gr.Interface(fn=chatbot,
                         inputs=gr.components.Textbox(lines=7, placeholder="Enter Question Related To Gita."),
                         outputs=" ",
                         title="Simple ChatGPT Bot with OpenAI API")
    iface.launch(share=True)
