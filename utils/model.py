'''
This module contains all the functions that controls the model creation, the functionalities, the tokenization and so on
'''
import torch
from LLaVA.llava.utils import disable_torch_init
from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.mm_utils import get_model_name_from_path
from LLaVA.llava.conversation import conv_templates

class chatModel:
    def __init__(self, model_path):
        '''
        Keep track of all the operations done on the model. 
        Inputs:
            model_path: the path of the model to load (in huggingface format)
        '''
        self.model_path = model_path
    
    def load_model(self, device):
        disable_torch_init()
        model_name = get_model_name_from_path(self.model_path)
        # Load the model 
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(self.model_path, None, model_name, load_8bit=False, load_4bit=True, device=device)
        
        # Load the conversation format
        if 'llama-2' in model_name.lower():
            self.conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            self.conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            self.conv_mode = "mpt"
        else:
            self.conv_mode = "llava_v0"

        self.device = device
        print("Model context length: ", self.context_len)
        print("Model loaded!")
    
    def remove_model(self):
        '''
        Function to remove the model from the memory
        '''
        del self.tokenizer
        del self.model
        del self.image_processor
        del self.context_len
        print("Model removed!")
    
    def start_chat(self):
        self.conversation = conv_templates[self.conv_mode].copy()
        
    def append_message(self, role, message):
        '''
        Function to append one message to the chat history. 
        Inputs:
            role: the role of the message sender (0 for user, 1 for model)
            message: the message to append
        '''
        self.conversation.append_message(role, message)
        
    def get_prompt(self):
        '''
        Function to get the current prompt
        '''
        return self.conversation.get_prompt()
    
    def generate(self, input_ids, image_tensor, temperature=0.9, max_new_tokens=100, stopping_criteria=None, streamer=None):
        
        image_tensor = image_tensor.to(self.device,dtype=torch.float16)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                streamer=streamer,
                stopping_criteria=[stopping_criteria])
    
        outputs = self.tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip().replace("\n", "")
        
        return outputs
    