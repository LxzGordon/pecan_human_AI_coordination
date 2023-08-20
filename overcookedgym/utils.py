import random
import time
from threading import Lock

import numpy as np
import torch
import openai

import tiktoken
from transformers import BertTokenizer, BertModel

from overcooked_ai_py.mdp.actions import Action, Direction


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in ['text-davinci-003']:
        converted_prompt = convert_messages_to_prompt(messages)
        return len(encoding.encode(converted_prompt))
    # elif model in ['gpt-3.5-turbo-0301', 'gpt-3.5-turbo', 'gpt-4', 'gpt-4-0314']:  # note: future models may deviate from this
    elif 'gpt' in model:
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")


def convert_messages_to_prompt(messages):
    """
    Converts a list of messages(for chat) to a prompt (for completion) for OpenAI's API.

    :param messages:
    :return: prompt
    """
    prompt = ""
    for message in messages:
        prompt += f"{message['content']}\n"

    return prompt


def gpt_state_list(mdp, state):
    """List representation of the current state, modified for GPT."""
    players_dict = {player.position: player for player in state.players}
    grid_list = []

    for y, terrain_row in enumerate(mdp.terrain_mtx):
        grid_row = []
        for x, element in enumerate(terrain_row):
            if (x, y) in players_dict.keys():
                player = players_dict[(x, y)]
                orientation = player.orientation
                player_object = player.held_object
                assert orientation in Direction.ALL_DIRECTIONS
                player_idx_lst = [
                    i
                    for i, p in enumerate(state.players)
                    if p.position == player.position
                ]
                assert len(player_idx_lst) == 1

                if player_object:
                    if player_object.name[0] == "s":
                        # this is a soup
                        grid_row.append("{}-{}-{}".format(player_idx_lst[0], Action.ACTION_TO_CHAR[orientation], str(player_object)))
                    else:
                        grid_row.append("{}-{}-{}".format(player_idx_lst[0], Action.ACTION_TO_CHAR[orientation], player_object.name[:1]))
                else:
                    grid_row.append("{}-{}".format(player_idx_lst[0], Action.ACTION_TO_CHAR[orientation]))
            else:
                if element == "X" and state.has_object((x, y)):
                    state_obj = state.get_object((x, y))
                    if state_obj.name[0] == "s":
                        grid_row.append(str(state_obj))
                    else:
                        grid_row.append(state_obj.name[:1])
                elif element == "P" and state.has_object((x, y)):
                    soup = state.get_object((x, y))
                    # display soup
                    grid_row.append(element+str(soup))
                else:
                    grid_row.append(element)

        grid_list.append(grid_row)

    if state.bonus_orders:
        bonus_orders = ["Bonus orders: {}".format(state.bonus_orders)]
        grid_list.append(bonus_orders)

    return grid_list


def retry_with_exponential_backoff(
        func,
        initial_delay: float = 1,
        exponential_base: float = 2,
        jitter: bool = True,
        max_retries: int = 10,
        errors: tuple = (openai.error.RateLimitError,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specific errors
            except errors as e:
                # Increment retries
                print(e)
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


def encode_text(self, text):
    '''get embdedding from text using BERT'''
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embeddings

# 计算相似度函数
def calculate_similarity(query_message, dialog_list:list):
    '''data = [{
        "role": "user",
        "content": user_input
        },
        {
        "role": "assistant",
        "content": assitant_result
    }
    ]'''
    query_embedding = encode_text(query_message["content"])
    data_embeddings = [encode_text(d["content"]) for d in dialog_list]
    similarity_scores = torch.nn.functional.cosine_similarity(query_embedding, torch.stack(data_embeddings), dim=1)
    return similarity_scores.numpy()

def get_closest_data(query_message, dialog_list:list, dia_num:int=10):
    similarity_scores = calculate_similarity(query_message, dialog_list)
    top_indices = np.argsort(similarity_scores)[-dia_num:][::-1]
    return [dialog_list[i] for i in top_indices]


class Node:
    """
        A node class for A* Pathfinding
        parent is parent of the current Node
        position is current position of the Node in the maze
        g is cost from start to current Node
        h is heuristic based estimated cost for current Node to end Node
        f is total cost of present node i.e. :  f = g + h
    """

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


def find_path(start_pos_and_or, other_pos_and_or, goal, terrain_mtx):
    start_node = Node(None, start_pos_and_or)
    end_node = Node(None, goal)

    yet_to_visit_list = []
    visited_list = []

    move = [(-1, 0),  # left
            (0, -1),  # up
            (1, 0),  # right
            (0, 1)]  # down

    n_rows = terrain_mtx['height']
    n_cols = terrain_mtx['width']
    mtx = terrain_mtx['matrix']

    mtx[other_pos_and_or[0][1]][other_pos_and_or[0][0]] = 'B'

    yet_to_visit_list.append(start_node)

    # let's just do bfs for now.
    while len(yet_to_visit_list) > 0:
        current_node = yet_to_visit_list[0]
        yet_to_visit_list.pop(0)
        visited_list.append(current_node)

        # 已经从一个方向到达，无需继续扩展了
        if current_node.position[0] == goal[0]:
            continue

        for new_position in move:
            node_position = (
                current_node.position[0][0] + new_position[0],
                current_node.position[0][1] + new_position[1]
            )

            # position out of bound
            if (node_position[0] > (n_cols - 1) or
                    node_position[0] < 0 or
                    node_position[1] > (n_rows - 1) or
                    node_position[1] < 0):
                continue

            if mtx[node_position[1]][node_position[0]] != ' ':
                continue

            new_node = Node(current_node, (node_position, new_position))

            if (new_node in visited_list) or (new_node in yet_to_visit_list):
                continue

            new_node.f = current_node.f + 1
            yet_to_visit_list.append(new_node)
            # visited
    last_node = None
    for i in visited_list:
        if i.position[0] == goal[0]:
            if last_node is None:
                if i.position[1] == goal[1]:
                    last_node = i
                else:
                    last_node = Node(i, (goal[0], goal[1]))
                    last_node.f = i.f + 1
            else:
                if i.position[1] == goal[1] and i.f < last_node.f:
                    last_node = i
                elif i.f + 1 < last_node.f:
                    last_node = Node(i, (goal[0], goal[1]))
                    last_node.f = i.f + 1

                    # no available plans.
    if last_node is None:
        return None, np.Inf
    else:
        print(f'planned last_node = {last_node.position}')
        previous_node = last_node
        while (previous_node.parent is not None) and (previous_node.parent != start_node):
            previous_node = previous_node.parent
            # already there.
        if previous_node == start_node:
            return Action.INTERACT, 1
        else:
            # did not move, changed direction
            if previous_node.position[0] == start_node.position[0]:
                return previous_node.position[1], last_node.f + 1
            else:
                # moved
                return (
                    previous_node.position[0][0] - start_node.position[0][0],
                    previous_node.position[0][1] - start_node.position[0][1]
                ), last_node.f + 1




class ThreadSafeSet(set):

    def __init__(self, *args, **kwargs):
        super(ThreadSafeSet, self).__init__(*args, **kwargs)
        self.lock = Lock()

    def add(self, *args):
        with self.lock:
            retval = super(ThreadSafeSet, self).add(*args)
        return retval

    def clear(self, *args):
        with self.lock:
            retval = super(ThreadSafeSet, self).clear(*args)
        return retval

    def pop(self, *args):
        with self.lock:
            if len(self):
                retval = super(ThreadSafeSet, self).pop(*args)
            else:
                retval = None
        return retval

    def remove(self, item):
        with self.lock:
            if item in self:
                retval = super(ThreadSafeSet, self).remove(item)
            else:
                retval = None
        return retval


class ThreadSafeDict(dict):

    def __init__(self, *args, **kwargs):
        super(ThreadSafeDict, self).__init__(*args, **kwargs)
        self.lock = Lock()

    def clear(self, *args, **kwargs):
        with self.lock:
            retval = super(ThreadSafeDict, self).clear(*args, **kwargs)
        return retval

    def pop(self, *args, **kwargs):
        with self.lock:
            retval = super(ThreadSafeDict, self).pop(*args, **kwargs)
        return retval

    def __setitem__(self, *args, **kwargs):
        with self.lock:
            retval = super(ThreadSafeDict, self).__setitem__(*args, **kwargs)
        return retval

    def __delitem__(self, item):
        with self.lock:
            if item in self:
                retval = super(ThreadSafeDict, self).__delitem__(item)
            else:
                retval = None
        return retval
