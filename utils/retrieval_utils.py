'''
This module serves as a container of functions needed for retrieval
'''

from sentence_transformers import SentenceTransformer
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity

def convert_dict_to_sentence(list_elements:List[Dict]):
    sentences = []
    sentence = ""
    for element in list_elements:
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
        sentence=""
    
    sentences = list(set(sentences)) # remove duplicates
    
    return sentences

def encode_information(list_elements):
    '''
    This function is called once per image, and encodes the information from openstreetmap using an embedding model
    '''
    model = SentenceTransformer('BAAI/bge-small-en-v1.5') # sentence-transformers/all-MiniLM-L6-v2
    sentences = convert_dict_to_sentence(list_elements)
    sentence_embeddings = model.encode(sentences, normalize_embeddings=True)
    del model
    
    return sentences, sentence_embeddings

def evaluate_similarity(query, sentences, elements_embeddings):
    '''
    Function to evaluate the similarity between a query and a list of sentences
    Inputs:
        query: the query
        sentences: the list of sentences
    Outputs:
        the list of similarities
    '''
    model = SentenceTransformer('BAAI/bge-small-en-v1.5') # sentence-transformers/all-MiniLM-L6-v2
    
    # Encode the query
    query_embedding = model.encode(query, normalize_embeddings=True).reshape(1, -1)
    # Calculate the cosine similarity
    similarities = query_embedding @ elements_embeddings.T
    similarities = list(similarities.squeeze())
    # Sort the similarities, keeping the index, to retrieve the elements 
    similarities = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
    print(similarities)
    
    # Return all the sentences that have a similarity of 0.4 or higher
    return [sentences[i[0]] for i in similarities if i[1]>=0.4]
