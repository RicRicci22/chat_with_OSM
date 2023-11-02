'''
This script launch a webapp that let you choose a bounding box and download the corresponding satellite image, then it starts the chat using llava model and OSM data. 
'''
import folium
import streamlit as st
import matplotlib.pyplot as plt
from folium.plugins import Draw
from streamlit_folium import st_folium

from utils.scrape import fetch_overpass_data, get_rbg_image, filter_osm_data, get_isolated_nodes, proj_lat_lon_on_image
from utils.model import chatModel
from utils.retrieval_utils import evaluate_similarity
from LLaVA.llava.mm_utils import tokenizer_image_token, tokenizer_image_token, KeywordsStoppingCriteria
from LLaVA.llava.mm_utils import process_images
from LLaVA.llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from transformers import TextStreamer

m = folium.Map(location=[39.949610, -75.150282], zoom_start=5)
Draw(export=True).add_to(m)

output = st_folium(m, width=700, height=500)

def get_bbox_bltr(bbox_coords):
    '''
    This function returns the bbox in the format [bottom, left, top, right]
    '''
    latitudes = [x[1] for x in bbox_coords[0]]
    longitudes = [x[0] for x in bbox_coords[0]]
    bottom = min(latitudes)
    top = max(latitudes)
    left = min(longitudes)
    right = max(longitudes)
    
    return [bottom, left, top, right]

# Let the user choose the model
model = st.sidebar.selectbox(
    'Choose the model',
    ('llava2',)
)

if 'controller' not in st.session_state:
    controller = chatModel(model_name="liuhaotian/llava-v1.5-7b")
    st.session_state['controller'] = controller

# Load the model
if st.sidebar.button('Load model'):
    device = "cuda:0"
    if model == 'llava2':
        if "controller" in st.session_state:
            st.session_state["controller"].load_model(device=device)


if st.button('Proceed'):
    last_bbox = output['last_active_drawing']
    
    bbox = get_bbox_bltr(last_bbox["geometry"]["coordinates"])
    
    # bottom, left, top, right = 46.04496229703382,10.98408579826355,46.04642930052508,10.986006259918213  # Parco Nadac, in Calavino, my place :)
    # bbox = (bottom, left, top, right)
    
    image = get_rbg_image(bbox) # PIL Image
    plt.imshow(image)
    plt.savefig("image.png")
    st.image(image)
    
    osm_data = fetch_overpass_data(bbox)
    
    osm_data = fetch_overpass_data(bbox)
    nodes, filtered = get_isolated_nodes(osm_data)
    located_elements = proj_lat_lon_on_image(bbox, filtered, nodes)
    #filtered_data = filter_osm_data(located_elements, elements_to_keep=["amenity", "building", "leisure"])
    
    # Start the chat by first describing the image 
    controller = st.session_state["controller"]
    
    controller.start_chat()
    prompt = "Give a detailed description of this satellite image."
    prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
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
    info_addition = "I will provide a list of informations about the area in the image. Each piece is enclosed by curly brakets. Use them to find the answer of the user's question, but don't reference to the pieces of information directly.\n"
    
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
        
        # Read the other question from CLI 
        # Retrieve the most similar "sentences"
        information = evaluate_similarity(inp, located_elements, k=5)
        prompt = info_addition 
        for info in information:
            prompt+="{"+info+"}"+"\n"
        prompt+=inp

        controller.append_message(controller.conversation.roles[0], prompt)
        controller.append_message(controller.conversation.roles[1], None)
        
        prompt = controller.get_prompt()
        
        print(prompt)
        
        input_ids = tokenizer_image_token(prompt, controller.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(controller.device)
        keywords = ["</s>"]
        stopping_criteria = KeywordsStoppingCriteria(keywords, controller.tokenizer, input_ids)
        temperature = 0.7
        max_new_tokens = 512
        
        out = controller.generate(input_ids, image_tensor, temperature=temperature, max_new_tokens=max_new_tokens, stopping_criteria=stopping_criteria, streamer=streamer)
        
        controller.conversation.messages[-1][-1] = out
        
        
