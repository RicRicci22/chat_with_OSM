'''
This module serves as a container of functions needed for retrieval
'''
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from typing import List, Dict


def count_buildings(list_elements:List[Dict]):
    '''
    This function counts the number of buildings in a list of elements
    '''
    count = 0
    for element in list_elements:
        #print(element)
        if "building" in element.keys():
            count+=1
    
    return "there are "+str(count)+" buildings in the image"

def convert_element_to_sentence(list_elements:List[Dict], llm=False):
    '''
    It is currently not working with llm = True
    '''
    sentences = []
    if llm:
        # Convert in sentence using a LLM (vicuna)
        # Load the model
        model_name_or_path = "TheBloke/vicuna-13B-v1.5-GPTQ"
        # To use a different branch, change revision
        # For example: revision="main"
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                    device_map="auto",
                                                    trust_remote_code=False,
                                                    revision="main")

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    
        for element in list_elements:
            prompt=f'''A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {str(element)} ASSISTANT:'''
            input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
            output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
            #print(tokenizer.decode(output[0]))
    else:
        for element in list_elements:
            sentence = ""
            for key, value in element.items():
                if "tiger" in key:
                    continue
                key = key.replace("_", " ")
                key = key.replace(":", " ")
                value = value.replace("_", " ")
                value = value.replace(":", " ")
                
                if key=="position":
                    sentence+="located in the "+str(value)+" of the image, "  
                else:
                    sentence+="the " + str(key)+" is "+str(value)+", "

            sentences.append(sentence[:-2])
    
    sentences = list(set(sentences)) # remove duplicates
    sentences.append(count_buildings(list_elements))
    
    return sentences

def encode_information(list_elements):
    '''
    This function is called once per image, and encodes the information from openstreetmap using an embedding model
    '''
    model = SentenceTransformer('BAAI/bge-small-en-v1.5') # sentence-transformers/all-MiniLM-L6-v2
    sentences = convert_element_to_sentence(list_elements, llm=False)
    sentence_embeddings = model.encode(sentences, normalize_embeddings=True)
    del model
    
    return sentences, sentence_embeddings

def evaluate_similarity(query, elements_textual, elements_embeddings):
    '''
    Function to evaluate the similarity between a query and a list of sentences
    Inputs:
        query: the query
        sentences: the list of sentences
    Outputs:
        the list of similarities
    '''
    model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1') # sentence-transformers/all-MiniLM-L6-v2
    
    # Encode the query
    query_embedding = model.encode(query, normalize_embeddings=True).reshape(1, -1)
    # Calculate the cosine similarity
    similarities = query_embedding @ elements_embeddings.T
    similarities = list(similarities.squeeze())
    # Sort the similarities, keeping the index, to retrieve the elements 
    similarities = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
    
    # Return the first 2 results
    #return [elements_textual[i[0]] for i in similarities[:2]]
    
    #Return all the sentences that have a similarity of 0.4 or higher
    return [elements_textual[i[0]] for i in similarities if i[1]>=0.4]


# https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models/ HAVE A LOOK AT THIS AFTER CONVERTING TO NATURAL LANGUAGE SENTENCES especially the passage on MULTI-QA