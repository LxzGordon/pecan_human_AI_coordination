import openai
from rich import print as rprint
import time
from typing import Union

from overcookedgym.utils import num_tokens_from_messages, convert_messages_to_prompt, retry_with_exponential_backoff
from overcookedgym.utils import get_closest_data



# Refer to https://platform.openai.com/docs/models/overview
TOKEN_LIMIT_TABLE = {
    "gpt-4": 8192,
    "gpt-4-0314": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-32k-0314": 32768,
    "gpt-3.5-turbo-16k": 16384,
    "gpt-3.5-turbo-0301": 4096,
    "gpt-3.5-turbo": 4096,
    "text-davinci-003": 4080
}


class GPTModule(object):
    """
    This module is responsible for communicating with GPTs.
    """
    def __init__(self, role_messages, 
                 model="gpt-3.5-turbo-0301",
                 retrival_method="recent_k",
                 K=10):
        '''
        args:

        use_similarity: 
        dia_num: the num of dia use need retrival from dialog history
        '''
        
        self.model = model
        self.retrival_method = retrival_method
        self.K = K

        self.chat_model = True if "gpt" in self.model else False
        # long term memory is permanent, and won't be erased by reset()
        self.instruction_head_list = role_messages
        self.dialog_history_list = []
        self.current_user_message = None
        self.cache_list = None

    def add_msgs_to_instruction_head(self, messages: Union[list, dict]):
        if isinstance(messages, list):
            self.instruction_head_list += messages
        elif isinstance(messages, dict):
            self.instruction_head_list += [messages]

    def add_msg_to_dialog_history(self, message: dict):
        self.dialog_history_list.append(message)
    
    def get_cache(self)->list:
        # 由于history里面成分负责，有input+query+sucess，只能根据input计算相似度，然后需要input+query的内容到
        # 失败存不存
        if self.retrival_method == "recent_k":
            if self.K > 0:
                return self.dialog_history_list[-self.K:]
            else: # dia_num == 0, means that we do not use in-context learning
                return []
        else:
             return get_closest_data(self.current_user_message, self.dialog_history_list, self.K)
            

    @property
    def query_messages(self):
        # return self.instruction_head_list + self.dialog_history_list
        return self.instruction_head_list + self.cache_list + [self.current_user_message]
    
    @property
    def prompt_token_length(self):
        return num_tokens_from_messages(self.query_messages, self.model)

    @retry_with_exponential_backoff
    def query(self, key, stop=None, temperature=0):
        openai.api_key = key
        self.cache_list = self.get_cache()
        self.restrict_dialogue()
        # messages = self.instruction_head_list + self.dialog_history_list
        messages = self.query_messages
        response = ""

        get_response = False
        retry_count = 0

        while not get_response:
            if retry_count > 3:
                rprint("[red][ERROR][/red]: Query GPT failed for over 3 times!")
                return {}
            try:
                if self.model in ['text-davinci-003']:
                    prompt = convert_messages_to_prompt(messages)
                    response = openai.Completion.create(
                        model=self.model,
                        prompt=prompt,
                        stop=stop,
                        temperature=temperature
                    )
                # elif self.model in ['gpt-3.5-turbo-0301', 'gpt-3.5-turbo', 'gpt-4', 'gpt-4-0314']:
                elif 'gpt' in self.model:
                    response = openai.ChatCompletion.create(
                        model=self.model,
                        messages=messages,
                        stop=stop,
                        temperature=temperature
                    )
                else:
                    # print(f"Model {self.model} not supported.")
                    # exit()
                    raise Exception(f"Model {self.model} not supported.")
                
                get_response = True
            except Exception as e:
                retry_count += 1
                rprint("[red][OPENAI ERROR][/red]:", e)
                time.sleep(20)

        return self.parse_response(response)

    def parse_response(self, response):
        if self.model in ['text-davinci-003']:
            return response["choices"][0]["text"]
        elif self.model in ['gpt-3.5-turbo-16k', 'gpt-3.5-turbo-0301', 'gpt-3.5-turbo', 'gpt-4', 'gpt-4-0314']:
            return response["choices"][0]["message"]["content"]

    def restrict_dialogue(self):
        """
        The limit on token length for gpt-3.5-turbo-0301 is 4096.
        If token length exceeds the limit, we will remove the oldest messages.
        """
        limit = TOKEN_LIMIT_TABLE[self.model]
        # TODO validate that the messages removed are obs and actions
        print(f'Current token: {self.prompt_token_length}')
        while self.prompt_token_length >= limit:
            # print(f"Restricting dialogue, messages removed:{self.state_action_prompts[0]}, {self.state_action_prompts[1]}")
            self.cache_list.pop(0)
            self.cache_list.pop(0)
            self.cache_list.pop(0)
            self.cache_list.pop(0)
            print(f'Update token: {self.prompt_token_length}')
            # print(f"New dialogue length: \
            # {num_tokens_from_messages(self.instruction_head_list + self.dialog_history_list , self.model)}/{limit}")

    def reset(self):
        self.dialog_history_list = []

