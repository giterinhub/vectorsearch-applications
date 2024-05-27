import os
import sys

from dotenv import load_dotenv, find_dotenv
from tiktoken import get_encoding
from weaviate.classes.query import Filter
from litellm import completion_cost
from loguru import logger 
import streamlit as st

# Append the parent directory to sys.path for the src imports to work
sys.path.append('../')

# Import custom modules
from src.database.weaviate_interface_v4 import WeaviateWCS
from src.database.database_utils import get_weaviate_client
from src.llm.llm_interface import LLM
from src.reranker import ReRanker
from src.llm.prompt_templates import generate_prompt_series, huberman_system_message
from app_functions import (convert_seconds, search_result, validate_token_threshold,
                           stream_chat, load_data)

# Load environment variables
load_dotenv(find_dotenv(), override=True)

# Page configuration
st.set_page_config(page_title="Huberman Labs - Discover Insights", 
                   page_icon=":microscope:", 
                   layout="wide", 
                   initial_sidebar_state="expanded"
                   )

# Custom CSS
st.markdown(
    """
    <style>
    .reportview-container {
        background: #f0f0f5;
    }
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set up app configuration
turbo = 'gpt-3.5-turbo-0125'
reader_model_name = turbo

data_path = '../data/huberman_labs.json'
embedding_model_path = '/workspaces/vectorsearch-applications/notebooks/models'

# Retrieve API credentials from environment variables
api_key = os.environ['WEAVIATE_API_KEY']
url = os.environ['WEAVIATE_ENDPOINT']

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

## QA MODEL
llm = LLM(model_name=reader_model_name)

## TOKENIZER
encoding = get_encoding("cl100k_base")

## Display properties
display_properties = ['guest', 'content', 'thumbnail_url', 'summary', 'title', 'episode_url', 'length_seconds', 'expanded_content']

# best practice is to dynamically load collections from weaviate using client.show_all_collections()
# available_collections = retriever.collections.list_all(simple=False) 
available_collections = retriever.show_all_collections()
logger.info(available_collections)

## COST COUNTER
if not st.session_state.get('cost_counter'):
    st.session_state['cost_counter'] = 0

def main(retriever: WeaviateWCS):
    # Load data
    try:
        data = load_data(data_path)
        #creates list of guests for sidebar
        guest_list = sorted(list(set([d['guest'] for d in data])))
    except FileNotFoundError:
        logger.error(f'Data file not found: {data_path}')
        st.error(f'Data file not found: {data_path}')
        return
    except Exception as e:
        logger.error(f'Error loading data: {e}')
        st.error(f'Error loading data. Please check the logs for more details.')
        return
    
    with st.sidebar:
        collection_name = st.selectbox('Collection Name', 
                                    options=available_collections, 
                                    index=1, 
                                    placeholder='Select Collection Name')

        guest_input = st.selectbox('Select Guest', 
                                options=guest_list, 
                                index=None, 
                                placeholder='Select Guest')

        alpha_input = st.slider('Alpha', min_value=0.0, max_value=1.0, value=0.54, step=0.01)
        retrieval_limit = st.slider('Retrieval Limit', min_value=1, max_value=200, value=10, step=1)
        reranker_topk = st.slider('Reranker TopK', min_value=1, max_value=10, value=3, step=1)
        temperature_input = st.slider('Temperature', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        verbosity = st.selectbox('Verbosity Level', options=[0, 1, 2], index=1)
        token_threshold = int(st.text_input('Token Threshold', value=2500))
        
    
    # retriever.return_properties.append('expanded_content')

    ##############################
    ##### SETUP MAIN DISPLAY #####
    ##############################
    st.image('./app_assets/hlabs_logo.png', width=400)
    st.image('./app_assets/64c461028c52bbd10ab5531d_about-bg.webp', width=500)
#    st.image(, width=800)
    st.subheader("Get to know Dr. Andrew :blue[Huberman] and his guests: :sunglasses:")
    st.write('\n')

    with st.form('my_form'):
        # text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
        query = st.text_input('Enter your question: ', help='You will be answered using the YouTube episodes from the Huberman Labs show.')
        submitted = st.form_submit_button('Search', help='Click to search for the answer to your question.')
        st.write('\n\n\n\n\n')

#    col1, _ = st.columns([7,3])
#    with col1:
#        query = st.text_input('Enter your question: ', help='WHat you enter here will be answered using the YouTube episodes from the Huberman Labs show hosted by Andrew Huberman.')
#        st.form_submit_button('Search', help='Click to search for the answer to your question.')
#        st.write('\n\n\n\n\n')
    #    if query:
    #        st.write('This app is not currently functioning as intended. Uncomment lines 100-172 to enable Q&A functionality.')
    ########################
    ##### SEARCH + LLM #####
    ########################
    if query and not collection_name:
        raise ValueError('Please first select a collection name')
    if query:
        # Validate token threshold
        if not len(encoding.encode(query))<=token_threshold/6:
            st.warning("Query exceeds token threshold. Please shorten your query or increase the token threshold value in the sidebar.")
            return
        # make hybrid call to weaviate
        guest_filter = Filter.by_property(name='guest').equal(guest_input) if guest_input else None
        # huberman = retriever.collections.get(collection_name)

        if guest_filter:
            logger.info(f'FILTER: {guest_filter}')
            hybrid_response = retriever.hybrid_search(query, collection_name, return_properties=display_properties, alpha=alpha_input, limit=retrieval_limit, filter=guest_filter)
        else:
            hybrid_response = retriever.hybrid_search(query, collection_name, return_properties=display_properties, alpha=alpha_input, limit=retrieval_limit)

        if not hybrid_response:
            st.warning("No results found.")
            return

        ranked_response = reranker.rerank(hybrid_response, query, top_k=reranker_topk)
        logger.info(f'# RANKED RESULTS: {len(ranked_response)}')   

        # token_threshold = 2500 # generally allows for 3-5 results of chunk_size 256
        content_field = 'content'
        # content_field = 'expanded_content'

        # validate token count is below threshold
        valid_response = validate_token_threshold(  ranked_response, 
                                                    query=query,
                                                    system_message=huberman_system_message,
                                                    tokenizer=encoding,# variable from ENCODING,
                                                    llm_verbosity_level=verbosity,
                                                    token_threshold=token_threshold, 
                                                    content_field=content_field,
                                                    verbose=True)
        logger.info(f'# VALID RESULTS: {len(valid_response)}')
        logger.info(f'# RESPONSE: {valid_response}')
        #set to False to skip LLM call
        make_llm_call = True
        # prep for streaming response
        with st.spinner('Generating Response...'):
            st.markdown("----")                
            # generate LLM prompt
            prompt = generate_prompt_series(query=query, results=valid_response, verbosity_level=verbosity)
            if make_llm_call:
                with st.chat_message('Huberman Labs', avatar='./app_assets/huberman_logo.png'):
                    stream_obj = stream_chat(llm, prompt, max_tokens=250, temperature=temperature_input)
                    string_completion = st.write_stream(stream_obj) # https://docs.streamlit.io/develop/api-reference/write-magic/st.write_stream
          
            # need to pull out the completion for cost calculation
            string_completion = ' '.join([c for c in stream_obj])
            call_cost = completion_cost(completion=string_completion, 
                                        model=turbo, 
                                        prompt=huberman_system_message + ' ' + prompt,
                                        call_type='completion')
            st.session_state['cost_counter'] += call_cost
            logger.info(f'TOTAL SESSION COST: {st.session_state["cost_counter"]}')

    # ##################
    # # SEARCH DISPLAY #
    # ##################
            st.subheader("Search Results")
            for i, hit in enumerate(valid_response):
                col1, col2 = st.columns([7, 3], gap='large')
                episode_url = hit['episode_url']
                title = hit['title']
                show_length = hit['length_seconds']
                # logger.info(f'show_length: {show_length} seconds')
                time_string = convert_seconds(show_length) # convert show_length to readable time string
                with col1:
                    st.write( search_result(i=i, 
                                            url=episode_url,
                                            guest=hit['guest'],
                                            title=title,
                                            content=ranked_response[i]['content']
                                            ,length=time_string
                                            ),
                                            unsafe_allow_html=True)
                    st.write('\n\n')

                with col2:
                    image = hit['thumbnail_url']
                    st.image(image, caption=title.split('|')[0], width=200, use_column_width=False)
                    st.markdown(f'<p style="text-align": right;"><b>Guest: {hit["guest"]}</b>', unsafe_allow_html=True)

if __name__ == '__main__':
    main(retriever)