import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from PIL import Image

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


def transform_elements_to_sentences(elements, llm=True):
    if llm:
        print("loading model..")
        model_name_or_path = "TheBloke/vicuna-13B-v1.5-GPTQ"
        # To use a different branch, change revision
        # For example: revision="main"
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                    device_map="auto",
                                                    trust_remote_code=False,
                                                    revision="main")

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        translated_elements={}
        for i in tqdm(range(len(elements))):
            element=elements[i]
            prompt=f'''A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {str(element)}. Translate this information from openstreetmap format into a simple sentence. Leave out unnecessary information for a human. Just provide the sentence, without adding thigs like "the sentence would be". ASSISTANT:'''
            input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
            output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
            translated_elements[i] = tokenizer.decode(output[0], skip_special_tokens=True).split("ASSISTANT:")[1].replace("\"", "").strip()
        
        # Save the translated elements
        with open("../translated_elements.json", "w") as f:
            json.dump(translated_elements, f, indent=4)
    else:
        pass

    
if __name__=="__main__":
    elements = json.load(open("../located_elements.json"))
    transform_elements_to_sentences(elements, llm=True)