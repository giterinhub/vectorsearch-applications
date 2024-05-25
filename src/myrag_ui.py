import os
import sys
from dotenv import load_dotenv, find_dotenv
import streamlit as st
from tiktoken import get_encoding
from weaviate.classes.query import Filter
from litellm import completion_cost
from loguru import logger

# Append the parent directory to sys.path
sys.path.append('../')

# Load environment variables
load_dotenv(find_dotenv(), override=True)

# Import custom modules
from src.database.weaviate_interface_v4 import WeaviateWCS
from src.database.database_utils import get_weaviate_client
from src.llm.llm_interface import LLM
from src.reranker import ReRanker
from src.llm.prompt_templates import generate_prompt_series, huberman_system_message
from app_functions import (convert_seconds, search_result, validate_token_threshold, 
                           stream_chat, load_data)

# Page configuration
st.set_page_config(page_title="Huberman Labs", layout="wide", initial_sidebar_state="auto")

# Set up app configuration
turbo = 'gpt-3.5-turbo-0125'
reader_model_name = turbo
data_path = '../data/huberman_labs.json'
embedding_model_path = '/workspaces/vectorsearch-applications/notebooks/models'

# Retrieve API credentials from environment variables
api_key = os.getenv('WEAVIATE_API_KEY')
url = os.getenv('WEAVIATE_ENDPOINT')

# Initialize Weaviate client
try:
    retriever = WeaviateWCS(endpoint=url, api_key=api_key, model_name_or_path=embedding_model_path)
    if retriever._client.is_live():
        logger.info('Weaviate is ready!')
except Exception as e:
    logger.error(f'Failed to connect to Weaviate: {e}')
    st.error('Failed to connect to Weaviate. Please check the logs for more details.')

# Initialize Reranker
try:
    reranker = ReRanker()
except Exception as e:
    logger.error(f'Failed to initialize Reranker: {e}')
    st.error('Failed to initialize Reranker. Please check the logs for more details.')

def main(retriever):
    st.title("Huberman Labs Search Interface")

    # Load data
    try:
        data = load_data(data_path)
    except FileNotFoundError:
        logger.error(f'Data file not found: {data_path}')
        st.error(f'Data file not found: {data_path}')
        return
    except Exception as e:
        logger.error(f'Error loading data: {e}')
        st.error(f'Error loading data. Please check the logs for more details.')
        return

    # User input for search
    query = st.text_input("Enter your search query:")
    
    if query:
        # Validate token threshold
        if not validate_token_threshold(query):
            st.warning("Query exceeds token threshold. Please shorten your query.")
            return

        try:
            # Retrieve data from Weaviate
            response = retriever.retrieve(query)
            if not response:
                st.warning("No results found.")
                return
            
            # Rerank the response
            ranked_response = reranker.rerank(response)
            
            # Display search results
            st.subheader("Search Results")
            for i, hit in enumerate(ranked_response):
                col1, col2 = st.columns([7, 3], gap='large')
                episode_url = hit.get('episode_url')
                title = hit.get('title')
                show_length = hit.get('length_seconds')
                time_string = convert_seconds(show_length)

                with col1:
                    st.write(
                        search_result(
                            i=i, 
                            url=episode_url,
                            guest=hit.get('guest'),
                            title=title,
                            content=hit.get('content'),
                            length=time_string
                        ),
                        unsafe_allow_html=True
                    )
                    st.write('\n\n')

                with col2:
                    image = hit.get('thumbnail_url')
                    st.image(image, caption=title.split('|')[0], width=200, use_column_width=False)
                    st.markdown(f'<p style="text-align: right;"><b>Guest: {hit.get("guest")}</b></p>', unsafe_allow_html=True)

        except Exception as e:
            logger.error(f'Error during retrieval or reranking: {e}')
            st.error(f'Error during retrieval or reranking. Please check the logs for more details.')

if __name__ == '__main__':
    main(retriever)
