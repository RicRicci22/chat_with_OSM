'''
This script launch a webapp that let you choose a bounding box and download the corresponding satellite image, then it starts the chat using llava model and OSM data. 
'''
import folium
import streamlit as st
from folium.plugins import Draw
from streamlit_folium import st_folium

from utils.scrape import fetch_overpass_data, get_rbg_image, get_pure_nodes, proj_lat_lon_on_image, filter_keys
from utils.general import get_bbox_bltr
from utils.model import chatModel
from utils.retrieval_utils import evaluate_similarity, encode_information
from LLaVA.llava.mm_utils import tokenizer_image_token, tokenizer_image_token, KeywordsStoppingCriteria
from LLaVA.llava.mm_utils import process_images
from LLaVA.llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from transformers import TextStreamer

map = st.sidebar.selectbox(
    'Choose the map layer',
    ('osm_map', 'satellite_map')
)

if map == 'osm_map':
    m = folium.Map(location=[39.949610, -75.150282], zoom_start=18)#, tiles='OpenStreetMap')
    Draw(export=True).add_to(m)
elif map == 'satellite_map':
    m = folium.Map(location=[39.949610, -75.150282], zoom_start=18)
    tile = folium.TileLayer(
            tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr = 'Esri',
            name = 'Esri Satellite',
            overlay = False,
            control = True
        ).add_to(m)
    
    Draw(export=True).add_to(m)

output = st_folium(m, width=700, height=500)

if 'controller' not in st.session_state:
    controller = chatModel(model_path="liuhaotian/llava-v1.5-7b")
    st.session_state['controller'] = controller
    device = "cuda:0"
    st.session_state["controller"].load_model(device=device)
    
# Let the user choose the model
model = st.sidebar.selectbox(
    'Choose the model',
    ('liuhaotian/llava-v1.5-7b',)
)

# Load the model
if st.sidebar.button('Load model'):
    if st.session_state['controller'].model_path != model:
        print("How")
        st.session_state['controller'].remove_model()
        st.session_state['controller'].model_path = model
        # Load the new model
        st.session_state["controller"].load_model(device=device)


if st.button('Proceed'):
    last_bbox = output['last_active_drawing']
    
    bbox = get_bbox_bltr(last_bbox["geometry"]["coordinates"])
    
    # bottom, left, top, right = 46.04496229703382,10.98408579826355,46.04642930052508,10.986006259918213  # Parco Nadac, in Calavino, my place :)
    # bbox = (bottom, left, top, right)
    
    image = get_rbg_image(bbox) # PIL Image
    st.image(image) # Show the image
    
    osm_data = fetch_overpass_data(bbox)
    
    nodes, filtered = get_pure_nodes(osm_data)
    located_elements = proj_lat_lon_on_image(bbox, filtered, nodes)
    located_elements = filter_keys(located_elements, ["source", "attribution", "massgis", "gnis"]) # manually remove.. mmmm
    for element in located_elements:
        print(element)
    elements_textual, elements_embeddings = encode_information(located_elements)
    #filtered_data = filter_osm_data(located_elements, elements_to_keep=["amenity", "building", "leisure"])
    
    # Start the chat by first describing the image 
    controller = st.session_state["controller"]
    
    controller.start_chat()
    
    # prompt="You are an assistant that can understand image contents and interact with me using text. I provided an image to you, for which I will sumbit queries. " \
    #         "You must analyze the query and decide if you can answer directly using your capabilities or if you need more information. " \
    #         "Since you are very good at understanding general concepts in the image, you can directly answer when the user asks for general information. However, if you are uncertain about the answer, you must ask for more information. " \
    #         "This is extremely useful when the user ask for specific information about the area in the image, such as the existance of particular structures (church, hospital, post office), names of the things in the image, addresses and so on. " \
    #         "To ask for more information you must reply just with the keyword 'get more data'. Let's start!"
    
    prompt = DEFAULT_IMAGE_TOKEN + '\n'
    controller.append_message(controller.conversation.roles[0], prompt)
    controller.append_message(controller.conversation.roles[1], None)
    prompt = controller.get_prompt()
    
    #print(prompt)
    
    input_ids = tokenizer_image_token(prompt, controller.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(controller.device)
    keywords = ["</s>"]
    stopping_criteria = KeywordsStoppingCriteria(keywords, controller.tokenizer, input_ids)
    temperature = 0.7
    max_new_tokens = 512
    
    model_cfg = dict()
    model_cfg["image_aspect_ratio"] = "pad"
    image_tensor = process_images([image], controller.image_processor, model_cfg=model_cfg)
    
    out = controller.generate(input_ids, image_tensor, temperature=temperature, max_new_tokens=max_new_tokens, stopping_criteria=stopping_criteria)
    
    controller.conversation.messages[-1][-1] = out
    
    # Append information of osm to the chat
    info_addition = "Before interacting with you, I provide some external information about the area in the image. Each piece is enclosed by curly brakets. " \
                    "If you find it useful to reply to my interrogation, use it. Reply concisely and clearly, following my interrogations. " \
    
    streamer = TextStreamer(controller.tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    while True:
        try:
            inp = input(f"{controller.conversation.roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break
        
        print(f"{controller.conversation.roles[1]}: ", end="")
        
        # Read the other question from CLI and retrieve the information
        information = evaluate_similarity(inp, elements_textual, elements_embeddings)
        
        # Insert info if necessary
        if len(information)!=0:
            prompt = info_addition + "\n"
            for info in information:
                prompt+="{"+info+"}"+"\n"
        prompt+= inp 
        
        # # Insert info if necessary
        # if len(information)!=0:
        #     prompt += info_addition + "\n"
        #     for info in information:
        #         prompt+="{"+info+"}"+"\n"
        
        #inp+=" If you need more information, reply with the keyword 'get more data'."
        controller.append_message(controller.conversation.roles[0], prompt)
        controller.append_message(controller.conversation.roles[1], None)
        
        prompt = controller.get_prompt()
        
        print(prompt)
        print("\n")
        
        input_ids = tokenizer_image_token(prompt, controller.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(controller.device)
        keywords = ["</s>"]
        stopping_criteria = KeywordsStoppingCriteria(keywords, controller.tokenizer, input_ids)
        temperature = 0.2
        max_new_tokens = 512
        
        out = controller.generate(input_ids, image_tensor, temperature=temperature, max_new_tokens=max_new_tokens, stopping_criteria=stopping_criteria, streamer=streamer)
        
        controller.conversation.messages[-1][-1] = out
        controller.conversation.messages[-2][-1] = inp 
        
        prompt=""
        

