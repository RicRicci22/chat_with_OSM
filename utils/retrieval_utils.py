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
            if "_" in key:
                key = key.replace("_", " ")
            if "_" in value:
                value = value.replace("_", " ")
              
            if key=="position":
                sentence+="located in the "+str(value)+" of the image, "  
            else:
                sentence+=str(key)+": "+str(value)+", "

        sentences.append(sentence[:-2])
        sentence=""
    
    return sentences
        

def evaluate_similarity(query, list_elements, k=None, prob=None):
    '''
    Function to evaluate the similarity between a query and a list of sentences
    Inputs:
        query: the query
        sentences: the list of sentences
    Outputs:
        the list of similarities
    '''
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    # Encode the query
    query_embedding = model.encode(query).reshape(1, -1)
    # Encode the elements
    sentences = convert_dict_to_sentence(list_elements)
    sentence_embeddings = model.encode(sentences)
    
    similarities = []
    for sentence in sentence_embeddings:
        sentence = sentence.reshape(1, -1)
        similarities.append(cosine_similarity(query_embedding, sentence))
    
    # Sort the similarities, keeping the index, to retrieve the elements 
    similarities = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
    
    if k!=None:
        # Return the sentences
        return [sentences[i[0]] for i in similarities[:k]]
    elif prob!=None:
        # Return all the sentences whose cumulative probability is greater than prob
        cumulative_prob = 0
        for i in similarities:
            cumulative_prob+=i[1]
            if cumulative_prob>=prob:
                return [sentences[i[0]] for i in similarities[:i[0]+1]]
    else:
        ValueError("You must specify either k or prob")
