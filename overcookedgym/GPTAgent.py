import itertools, os, json, re
from collections import defaultdict
import numpy as np
import pkg_resources
import copy 
from overcookedgym.GPTModules import GPTModule
from overcooked_ai_py.mdp.actions import Action, Direction
from overcookedgym.utils import find_path

cwd = os.getcwd()
# openai_key_file = os.path.join(cwd, "openai_key.txt")
openai_key_file = "overcookedgym/openai_key.txt"
# layout_prompt_dir = os.path.join(cwd, "prompts/layout")
# PROMPT_DIR = os.path.join(cwd, "prompts")
PROMPT_DIR = "overcookedgym/prompts"

NAME_TO_ACTION = {
    "NORTH": Direction.NORTH,
    "SOUTH": Direction.SOUTH,
    "EAST": Direction.EAST,
    "WEST": Direction.WEST,
    "INTERACT": Action.INTERACT,
    "STAY": Action.STAY
}


class GPTAgent(object):
    """
    This agent uses GPT-3.5 to generate actions.
    """
    def __init__(self, model="gpt-3.5-turbo-0301"):
        self.agent_index = None
        self.model = model

        self.openai_api_keys = []
        self.load_openai_keys()
        self.key_rotation = True

    def load_openai_keys(self):
        with open(openai_key_file, "r") as f:
            context = f.read()
        self.openai_api_keys = context.split('\n')

    def openai_api_key(self):
        if self.key_rotation:
            self.update_openai_key()
        return self.openai_api_keys[0]

    def update_openai_key(self):
        self.openai_api_keys.append(self.openai_api_keys.pop(0))

    def set_agent_index(self, agent_index):
        raise NotImplementedError

    def action(self, state):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class GPTMediumLevelAgent(GPTAgent):
    """
    This agent uses GPT-3.5 to generate medium level actions.
    """
    def __init__(
            self,
            mlam,
            layout,
            auto_unstuck=True,
            model='gpt-3.5-turbo-0301',
            retrival_method="recent_k",
            K=10, 
    ):
        super().__init__(model=model)
        self.mlam = mlam
        self.layout = layout
        self.mdp = self.mlam.mdp

        # self.retrival_method = retrival_method
        # self.K = K

        self.planner = self.create_gptmodule("planner", retrival_method=retrival_method, K=K)
        # self.cooperator = self.create_gptmodule("cooperator")
        self.explainer = self.create_gptmodule("explainer", retrival_method='recent_k', K=0)
        # self.parser = self.create_gptmodule("parser")
        # self.replanner = self.create_gptmodule("replanner")
        # self.validator = self.create_gptmodule("validator")

        # Whether to automatically take an action to get the agent uPlayer{self.agent_index} in the same
        # state as the previous turn. If false, the agent is history-less, while if true it has history.
        self.prev_state = None
        self.auto_unstuck = auto_unstuck

        self.current_ml_action = None
        self.current_ml_action_steps = 0
        self.time_to_wait = 0
        self.possible_motion_goals = None
        self.pot_id_to_pos = []

        self.layout_prompt = self.generate_layout_prompt()
        # layout_messages = [{"role": "user", "content": self.layout_prompt}]
        # self.planner.add_msgs_to_instruction_head(layout_messages)
        print(self.planner.instruction_head_list[0]['content'])

    def set_mdp(self, mdp):
        self.mdp = mdp

    def create_gptmodule(self, module_name, file_type='txt', retrival_method='recent_k', K=10):
        print("Initializing GPT {}.".format(module_name))
        # prompt_file = os.path.join(prompts_dir, self.layout, module_name, self.model+'.'+file_type)
        prompt_file = os.path.join(PROMPT_DIR, module_name, self.layout+'.'+file_type)
        with open(prompt_file, "r") as f:
            if file_type == 'json':
                messages = json.load(f)
            elif file_type == 'txt':
                messages = [{"role": "system", "content": f.read()}]
            else:
                print("Unsupported file format.")
        return GPTModule(messages, self.model, retrival_method, K)

    def reset(self):
        # self.agent_index = None
        self.planner.reset()
        self.explainer.reset()
        self.prev_state = None
        self.current_ml_action = None
        self.current_ml_action_steps = 0
        self.time_to_wait = 0
        self.possible_motion_goals = None
        # self.pot_id_to_pos = []

    def set_agent_index(self, agent_index):
        self.agent_index = agent_index
        
    def generate_layout_prompt(self):
        layout_prompt_dict = {
            # "onion_dispenser": " onion_dispenser {id}",
            # "dish_dispenser": " dish_dispenser {id}",
            # "tomato_dispenser": " tomato_dispenser {id}",
            # "serving": " serving_loc {id}",
            # "pot": " pot {id}",
            "onion_dispenser": " <Onion Dispenser {id}>",
            "dish_dispenser": " <Dish Dispenser {id}>",
            "tomato_dispenser": " <Tomato Dispenser {id}>",
            "serving": " <Serving Loc {id}>",
            "pot": " <Pot {id}>",
            # "counter": "<counter,{pos}>"
            # "onion_dispenser": " <onion  dispenser {id}> at {pos}",
            # "dish_dispenser": " <dish dispenser {id}> at {pos}",
            # "tomato_dispenser": " <tomato dispenser {id}> at {pos}",
            # "serving": " <serving loc {id}> at {pos}",
            # "pot": " <pot {id}> at {pos}",
            # "counter": " <counter {id}>, at {pos}"
        }


        layout_prompt = "Here's the layout of the kitchen:"
        for obj_type, prompt_template in layout_prompt_dict.items():
            locations = getattr(self.mdp, f"get_{obj_type}_locations")()
            for obj_id, obj_pos in enumerate(locations):
                # layout_prompt += prompt_template.format(id=obj_id, pos=pos) + ","
                layout_prompt += prompt_template.format(id=obj_id) + ","
                if obj_type == "pot":
                    self.pot_id_to_pos.append(obj_pos)
            # else:
            #     for pos in locations:
            #         layout_prompt += prompt_template.format(pos=pos) + ","
        layout_prompt = layout_prompt[:-1] + "."
        return layout_prompt

    def generate_state_prompt(self, state):

        player = state.players[self.agent_index]
        teammate = state.players[1 - self.agent_index]

        # next_order = list(state.all_orders)[0]
        # order_prompt = f"Current order: {next_order.ingredients}. "
        # next_order = list(state.order_list)[0]
        # order_prompt = f''

        time_prompt = f"Scene {state.timestep}: "
        # time_prompt = f"Current timestep: {state.timestep}. "

        # player_direction_str = Direction.DIRECTION_TO_NAME[player.orientation]
        # teammate_direction_str = Direction.DIRECTION_TO_NAME[teammate.orientation]
        ego_object = player.held_object.name if player.held_object else "nothing"
        teammate_object = teammate.held_object.name if teammate.held_object else "nothing"
        # ego_state_prompt = f"I am holding {ego_object}. "
        # teammate_state_prompt = f"Teammate is holding {teammate_object}. "
        # ego_state_prompt = f"<Player {self.agent_index}> holds "
        ego_state_prompt = f"Player {self.agent_index} holds "
        if ego_object == 'soup':
            ego_state_prompt += f"a dish with {ego_object} and needs to deliver soup.  "
        elif ego_object == 'nothing':
            ego_state_prompt += f"{ego_object}. "
        else:
            ego_state_prompt += f"one {ego_object}. "
        
        # teammate_state_prompt = f"<Player {1-self.agent_index}> holds "
        teammate_state_prompt = f"Player {1-self.agent_index} holds "
        if teammate_object == 'soup':
            teammate_state_prompt += f"a dish with {teammate_object} and needs to deliver soup. "
        elif teammate_object == "nothing":
            teammate_state_prompt += f"{teammate_object}. "
        else:
            teammate_state_prompt += f"one {teammate_object}. "

        
        kitchen_state_prompt = "Kitchen states: "
        # TODO add real ingredients that exists in pot
        prompt_dict = {
            # "empty": "pot_{id} is empty; ",
            # "cooking": "soup in pot_{id} is cooking, it will be ready after {t} timesteps; ",
            # "ready": "soup in pot_{id} is ready; ",
            # "1_items": "pot_{id} has 1 onion; ",
            # "2_items": "pot_{id} has 2 onions; ",
            # "3_items": "pot_{id} has 3 onions and is full; "
            "empty": "<Pot {id}> is empty; ",
            "cooking": "<Pot {id}> start cooking, the soup will be ready after {t} timesteps; ",
            # "ready": "soup in <Pot {id}> is ready; ",
            "ready": "<Pot {id}> has already cooked the soup and soup needs to be filled in a dish; ",
            "1_items": "<Pot {id}> has 1 onion; ",
            "2_items": "<Pot {id}> has 2 onions; ",
            "3_items": "<Pot {id}> has 3 onions and is full; "
        }

        pot_states_dict = self.mdp.get_pot_states(state)
        if pkg_resources.get_distribution("overcooked_ai").version == '1.1.0':
            '''pot_states_dict = {
            'empty': [positions of empty pots]
            'x_items': [soup objects with x items that have yet to start cooking],
            'cooking': [soup objs that are cooking but not ready]
            'ready': [ready soup objs],
            }'''
            for key in pot_states_dict.keys():
                if key == "cooking":
                    for pos in pot_states_dict[key]:
                        pot_id = self.pot_id_to_pos.index(pos)
                        soup_object = state.get_object(pos)
                        kitchen_state_prompt += prompt_dict[key].format(id=pot_id, t=soup_object.cook_time_remaining)
                else:
                    for pos in pot_states_dict[key]:
                        pot_id = self.pot_id_to_pos.index(pos)
                        kitchen_state_prompt += prompt_dict[key].format(id=pot_id)
            
        elif pkg_resources.get_distribution("overcooked_ai").version == '0.0.1':
            """pot_states_dict:
            {
            empty: [ObjStates]
            onion: {
                'x_items': [soup objects with x items],
                'cooking': [ready soup objs]
                'ready': [ready soup objs],
                'partially_full': [all non-empty and non-full soups]
                }
            tomato: same dict structure as above 
            }
            """
            for key in pot_states_dict.keys():
                if key == "empty":
                    for pos in pot_states_dict[key]:
                        pot_id = self.pot_id_to_pos.index(pos)
                        kitchen_state_prompt += prompt_dict[key].format(id=pot_id)
                else: # key = 'onion' or 'tomota'
                    for soup_key in pot_states_dict[key].keys():
                        # soup_key: ready, cooking, partially_full
                        for pos in pot_states_dict[key][soup_key]:
                            pot_id = self.pot_id_to_pos.index(pos)
                            soup_object = state.get_object(pos)
                            soup_type, num_items, cook_time = soup_object.state
                            if soup_key == "cooking":
                                kitchen_state_prompt += prompt_dict[soup_key].format(id=pot_id, t=self.mdp.soup_cooking_time-cook_time)
                            elif soup_key == "partially_full":
                                pass
                            else:
                                kitchen_state_prompt += prompt_dict[soup_key].format(id=pot_id)

        # kitchen_state_prompt = kitchen_state_prompt[:-1]
        # TODO add counter objects

        # return time_prompt + order_prompt + ego_state_prompt + teammate_state_prompt + kitchen_state_prompt
        return time_prompt + ego_state_prompt + teammate_state_prompt + kitchen_state_prompt

    def action(self, state):
        start_pos_and_or = state.players_pos_and_or[self.agent_index]

        # if current ml action does not exist, generate a new one
        if self.current_ml_action is None:
            self.current_ml_action = self.generate_ml_action(state)

        # if the current ml action is in process, Player{self.agent_index} done, else generate a new one
        if self.current_ml_action_steps > 0:
            current_ml_action_done = self.check_current_ml_action_done(state)
            if current_ml_action_done:
                # generate a new ml action
                self.generate_success_feedback(state)
                self.current_ml_action = self.generate_ml_action(state)

        count = 0
        while not self.validate_current_ml_action(state):
            # print(f"Player {self.agent_index}: {self.current_ml_action} is invalid.Generating a new one, trial {count}")
            self.generate_failure_feedback(state)
            
            self.current_ml_action = self.generate_ml_action(state)
            count += 1
            if count > 2:
                # raise Exception("Failed to generate a valid ml action")
                self.current_ml_action = "wait(1)"
                self.time_to_wait = 1
                self.planner.dialog_history_list.pop() # delete this analysis 
                wait_response = f'Analysis: the low-level controller meets problem. Let wait one step for a while.\nPlan for Player {self.agent_index}: "wait(1)".'
                self.planner.add_msg_to_dialog_history({"role": "assistant", "content": f'{wait_response}'})

        if "wait" in self.current_ml_action:
            self.current_ml_action_steps += 1
            self.time_to_wait -= 1
            chosen_action = Action.STAY
            if pkg_resources.get_distribution("overcooked_ai").version == '1.1.0':
                return chosen_action, {}
            elif pkg_resources.get_distribution("overcooked_ai").version == '0.0.1':
                return chosen_action
        
        else:
            possible_motion_goals = self.find_motion_goals(state)
            current_motion_goal, chosen_action = self.choose_motion_goal(
                start_pos_and_or, possible_motion_goals, state=state)       

            print(f'chosen action = {Action.to_char(chosen_action)}')    
        
        if self.auto_unstuck:
            # print(f"There they auto_unstuck")
            # HACK: if two agents get stuck, select an action at random that would
            # change the player positions if the other player were not to move
            if (
                    self.prev_state is not None
                    and state.players_pos_and_or
                    == self.prev_state.players_pos_and_or
            ):
                if self.agent_index == 0:
                    joint_actions = list(
                        itertools.product(Action.ALL_ACTIONS, [Action.STAY])
                    )
                elif self.agent_index == 1:
                    joint_actions = list(
                        itertools.product([Action.STAY], Action.ALL_ACTIONS)
                    )
                else:
                    raise ValueError("Player index not recognized")

                unblocking_joint_actions = []
                for j_a in joint_actions:
                    if pkg_resources.get_distribution("overcooked_ai").version == '1.1.0':
                        new_state, _ = self.mlam.mdp.get_state_transition(state, j_a)
                    elif pkg_resources.get_distribution("overcooked_ai").version == '0.0.1':
                        new_state, _, _ = self.mlam.mdp.get_state_transition(state, j_a)
                    
                    if (
                            new_state.player_positions
                            != self.prev_state.player_positions
                    ):
                        unblocking_joint_actions.append(j_a)
                # Getting stuck became a possibility simply because the nature of a layout (having a dip in the middle)
                if len(unblocking_joint_actions) == 0:
                    unblocking_joint_actions.append([Action.STAY, Action.STAY])
                chosen_action = unblocking_joint_actions[
                    np.random.choice(len(unblocking_joint_actions))
                ][self.agent_index]

            # NOTE: Assumes that calls to the action method are sequential
            self.prev_state = state

        
        if chosen_action is None:
            self.current_ml_action = "wait(1)"
            self.time_to_wait = 1
            chosen_action = Action.STAY
        
        self.current_ml_action_steps += 1
        

        print(f'P{self.agent_index} : {Action.to_char(chosen_action)}')    
        
        # print('P{}'.format(self.))
        if pkg_resources.get_distribution("overcooked_ai").version == '1.1.0':
            return chosen_action, {}
        elif pkg_resources.get_distribution("overcooked_ai").version == '0.0.1':
            return chosen_action

    def generate_ml_action(self, state):
        """
        Selects a medium level action for the current state.
        Motion goals can be thought of instructions of the form:
            [do X] at location [Y]

        In this method, X (e.g. deliver the soup, pick up an onion, etc) is chosen based on
        a simple set of greedy heuristics based on the current state.

        Effectively, will return a list of all possible locations Y in which the selected
        medium level action X can be performed.
        """
        state_prompt = self.generate_state_prompt(state)
        # planner_question = " Which action should I take?"
        # planner_question = " Which skill should I take?" 
        # state_prompt += planner_question
        print(f"\n+++++++++++++ Observation module to GPT+++++++++++++")
        print(f"{state_prompt}")

        state_message = {"role": "user", "content": state_prompt}
        self.planner.current_user_message = state_message
        response = self.planner.query(key=self.openai_api_key(), stop='Scene')
        # response = response.lower()
        
        # this message must be add into history after query
        self.planner.add_msg_to_dialog_history(state_message) 
        self.planner.add_msg_to_dialog_history({"role": "assistant", "content": response})
        
        print(f"\n=============GPT Planner module============")
        print(f'======query')
        print(response)

        
        # action_string = response
        # action_string = input('choose action: ')


        # Parse the response to get the medium level action string
        # ml_action = action_string.split()[0]
        pattern = r'layer\s*0: "(.+?)"'
        match = re.search(pattern, response)
        if match:
            action_string = match.group(1)
        else:
            # raise Exception("please check the query")
            print("please check the query")
            action_string = "wait(1)"
        
        # if "pickup" in action_string:
        #     if "onion" in action_string:
        #         ml_action = "pickup_onion"
        #     elif "tomato" in action_string:
        #         ml_action = "pickup_tomato"
        #     elif "dish" in action_string:
        #         ml_action = "pickup_dish"
        # elif "put" in action_string:
        #     if "onion" in action_string:
        #         ml_action = "put_onion_in_pot"
        #     elif "tomato" in action_string:
        #         ml_action = "put_tomato_in_pot"
        # elif "fill" in action_string:
        #     # fill_dish_with_soup in prompt
        #     ml_action = "fill_dish_with_soup"
        # elif "place" in action_string:
        #     ml_action = "place_obj_on_counter"
        # elif "start" in action_string:
        #     ml_action = "start_cooking"
        # elif "deliver" in action_string:
        #     ml_action = "deliver_soup"
        if "wait" in action_string:
            def parse_wait_string(s):
                # Check if it's just "wait"
                if s == "wait":
                    return 1

                # Remove 'wait' and other characters from the string
                s = s.replace('wait', '').replace('(', '').replace(')', '')

                # If it's a number, return it as an integer
                if s.isdigit():
                    return int(s)

                # If it's not a number, return a default value or raise an exception
                return 1
            self.time_to_wait = parse_wait_string(action_string)
            ml_action = f"wait({self.time_to_wait})"

        else:
            ml_action = action_string
        
        # aviod to generate two skill, eg, Plan for Player 0: "deliver_soup(), pickup(onion)".
        if "," in ml_action:
            ml_action = ml_action.split(',')[0].strip()


        print(f'\n======parser')
        print(f"Player {self.agent_index}: {ml_action}")

        self.current_ml_action_steps = 0

        return ml_action

    def check_current_ml_action_done(self,state):
        """
        checks if the current ml action is done
        :return: True or False
        """
        player = state.players[self.agent_index]
        pot_states_dict = self.mlam.mdp.get_pot_states(state)

        if "pickup" in self.current_ml_action:
            # obj_str = self.current_ml_action.split("_")[1]
            # obj_str = self.current_ml_action.split('(')[1][:-1]
            # pattern = r"pickup[\(_]([^\)_]+)[\)_]" 
            # pattern = r"pickup[_](\w+)"
            pattern = r"pickup(?:[(]|_)(\w+)(?:[)]|)" # fit both pickup(onion) and pickup_onion
            obj_str = re.search(pattern, self.current_ml_action).group(1)
            return player.has_object() and player.get_object().name == obj_str
        
        elif "fill" in self.current_ml_action:
            return player.held_object.name == 'soup'
        
        elif "put" in self.current_ml_action or "place" in self.current_ml_action:
            # TODO need to ensure that the object is placed at the right place(pot,counter)
            return not player.has_object()
        
        # elif "start" in self.current_ml_action:
        #     # player should be facing a pot and the pot is lit
        #     # TODO need to ensure that the player is facing the pot it was going to light up
        #     player_facing_position = (player.position[0] + player.orientation[0],
        #                               player.position[1] + player.orientation[1])
        #     if pkg_resources.get_distribution("overcooked_ai").version == '1.1.0':
        #         return player_facing_position in pot_states_dict["cooking"]
        #     elif pkg_resources.get_distribution("overcooked_ai").version == '0.0.1':
        #         return player_facing_position in pot_states_dict["onion"]["cooking"] or player_facing_position in pot_states_dict["tomato"]["cooking"]
        
        elif "deliver" in self.current_ml_action:
            return not player.has_object()
        
        elif "wait" in self.current_ml_action:
            # return self.wait_time >= int(self.current_ml_action.split('(')[1][:-1])
            return self.time_to_wait == 0

    def validate_current_ml_action(self, state):
        """
        make sure the current_ml_action exists and is valid
        """
        if self.current_ml_action is None:
            return False

        pot_states_dict = self.mdp.get_pot_states(state)
        player = state.players[self.agent_index]
        if pkg_resources.get_distribution("overcooked_ai").version == '1.1.0':
            soup_cooking = len(pot_states_dict['cooking']) > 0
            soup_ready = len(pot_states_dict['ready']) > 0
            pot_not_full = pot_states_dict["empty"] + self.mdp.get_partially_full_pots(pot_states_dict)
            cookable_pots = self.mdp.get_full_but_not_cooking_pots(pot_states_dict)
        elif pkg_resources.get_distribution("overcooked_ai").version == '0.0.1':
            soup_cooking = len(pot_states_dict['onion']['cooking'])+len(pot_states_dict['tomato']['cooking']) > 0
            soup_ready = len(pot_states_dict['onion']['ready'])+len(pot_states_dict['tomato']['ready']) > 0
            pot_not_full = pot_states_dict["empty"] + pot_states_dict["onion"]['partially_full'] + pot_states_dict["tomato"]['partially_full']
            cookable_pots = pot_states_dict["onion"]['{}_items'.format(self.mdp.num_items_for_soup)] + pot_states_dict["tomato"]['{}_items'.format(self.mdp.num_items_for_soup)] # pot has max onions/tomotos

        
        has_onion = False
        has_tomato = False
        has_dish = False
        has_soup = False
        has_object = player.has_object()
        if has_object:
            has_onion = player.get_object().name == 'onion'
            has_tomato = player.get_object().name == 'tomato'
            has_dish = player.get_object().name == 'dish'
            has_soup = player.get_object().name == 'soup'
        empty_counter = self.mdp.get_empty_counter_locations(state)

        if self.current_ml_action in ["pickup(onion)", "pickup_onion"]:
            return not has_object and len(self.mdp.get_onion_dispenser_locations()) > 0
        if self.current_ml_action in ["pickup(tomato)", "pickup_tomato"]:
            return not has_object and len(self.mdp.get_tomato_dispenser_locations()) > 0
        elif self.current_ml_action in ["pickup(dish)", "pickup_dish"]:
            return not has_object and len(self.mdp.get_dish_dispenser_locations()) > 0
        elif "put_onion_in_pot" in self.current_ml_action:
            return has_onion and len(pot_not_full) > 0
        elif "put_tomato_in_pot" in self.current_ml_action:
            return has_tomato and len(pot_not_full) > 0
        # elif self.current_ml_action == "place_obj_on_counter(dish)":
        #     return has_dish and len(empty_counter) > 0
        # elif self.current_ml_action == "place_obj_on_counter(onion)":
        #     return has_onion and len(empty_counter) > 0
        # elif self.current_ml_action == "place_obj_on_counter(tomato)":
        #     return has_tomato and len(empty_counter) > 0
        elif "place_obj_on_counter" in self.current_ml_action:
            return has_object and len(empty_counter) > 0
        # elif "start_cooking" in self.current_ml_action:
        #     return not has_object and len(cookable_pots) > 0
            # return not has_object
        elif "fill_dish_with_soup" in self.current_ml_action:
            return has_dish and (soup_ready or soup_cooking)
        elif "deliver_soup" in self.current_ml_action:
            return has_soup
        elif "wait" in self.current_ml_action:
            # return True
            return 0 < int(self.current_ml_action.split('(')[1][:-1]) <= 20
        # return False

    def find_motion_goals(self, state):
        """
        Generates the motion goals for the given medium level action.
        :param state:
        :return:
        """
        am = self.mlam
        motion_goals = []
        player = state.players[self.agent_index]
        pot_states_dict = self.mdp.get_pot_states(state)
        counter_objects = self.mdp.get_counter_objects_dict(
            state, list(self.mdp.terrain_pos_dict["X"])
        )
        if self.current_ml_action in ["pickup(onion)", "pickup_onion"]:
            motion_goals = am.pickup_onion_actions(state, counter_objects)
        elif self.current_ml_action in ["pickup(tomato)", "pickup_tomato"]:
            motion_goals = am.pickup_tomato_actions(state, counter_objects)
        elif self.current_ml_action in ["pickup(dish)", "pickup_dish"]:
            motion_goals = am.pickup_dish_actions(state, counter_objects)
        elif "put_onion_in_pot" in self.current_ml_action:
            motion_goals = am.put_onion_in_pot_actions(pot_states_dict)
        elif "put_tomato_in_pot" in self.current_ml_action:
            motion_goals = am.put_tomato_in_pot_actions(pot_states_dict)
        elif "place_obj_on_counter" in self.current_ml_action:
            motion_goals = am.place_obj_on_counter_actions(state)
        elif "start_cooking" in self.current_ml_action:
            if pkg_resources.get_distribution("overcooked_ai").version == '1.1.0':
                next_order = list(state.all_orders)[0]
                soups_ready_to_cook_key = "{}_items".format(len(next_order.ingredients))
                soups_ready_to_cook = pot_states_dict[soups_ready_to_cook_key]
            elif pkg_resources.get_distribution("overcooked_ai").version == '0.0.1':
                soups_ready_to_cook = pot_states_dict["onion"]['{}_items'.format(self.mdp.num_items_for_soup)] + pot_states_dict["tomato"]['{}_items'.format(self.mdp.num_items_for_soup)]
            only_pot_states_ready_to_cook = defaultdict(list)
            only_pot_states_ready_to_cook[soups_ready_to_cook_key] = soups_ready_to_cook
            motion_goals = am.start_cooking_actions(only_pot_states_ready_to_cook)
        elif "fill_dish_with_soup" in self.current_ml_action:
            motion_goals = am.pickup_soup_with_dish_actions(pot_states_dict, only_nearly_ready=True)
        elif "deliver_soup" in self.current_ml_action:
            motion_goals = am.deliver_soup_actions()
        elif "wait" in self.current_ml_action:
            motion_goals = am.wait_actions(player)
        else:
            raise ValueError("Invalid action: {}".format(self.current_ml_action))

        motion_goals = [
            mg
            for mg in motion_goals
            if self.mlam.motion_planner.is_valid_motion_start_goal_pair(
                player.pos_and_or, mg
            )
        ]

        return motion_goals

    def choose_motion_goal(self, start_pos_and_or, motion_goals, state = None):
        """
        For each motion goal, consider the optimal motion plan that reaches the desired location.
        Based on the plan's cost, the method chooses a motion goal (either boltzmann rationally
        or rationally), and returns the plan and the corresponding first action on that plan.
        """
        (
            chosen_goal,
            chosen_goal_action,
        ) = self.get_lowest_cost_action_and_goal_new(
            start_pos_and_or, motion_goals, state
        )
        return chosen_goal, chosen_goal_action
    

    def real_time_planner(self, start_pos_and_or, goal, state):   
        terrain_matrix = {
            'matrix': copy.deepcopy(self.mlam.mdp.terrain_mtx), 
            'height': len(self.mlam.mdp.terrain_mtx), 
            'width' : len(self.mlam.mdp.terrain_mtx[0]) 
        }
        other_pos_and_or = state.players_pos_and_or[1 - self.agent_index]
        action_plan, plan_cost = find_path(start_pos_and_or, other_pos_and_or, goal, terrain_matrix) 

        return action_plan, plan_cost
    
    def get_lowest_cost_action_and_goal_new(self, start_pos_and_or, motion_goals, state): 
        """
        Chooses motion goal that has the lowest cost action plan.
        Returns the motion goal itself and the first action on the plan.
        """   
        min_cost = np.Inf
        best_action, best_goal = None, None
        for goal in motion_goals:   
            """
                print(start_pos_and_or, goal) 
                ((4, 3), (0, -1)) ((3, 3), (0, 1))
                sys.exit(0) 
            """
            # MARKING: 通过 self.mlam.motion_planner.get_plan 计算出这个 plan 的 cost 
            action_plan, plan_cost = self.real_time_planner(
                start_pos_and_or, goal, state
            )     

            #print(start_pos_and_or, goal)  

            #sys.exit(0) 
            """
            print(action_plan)  = [(-1, 0), (0, 1), 'interact']
            sys.exit(0) 
            """
            if plan_cost < min_cost:
                best_action = action_plan
                min_cost = plan_cost
                best_goal = goal   
        #print(best_goal, best_action) 
        #sys.exit(0) 
        """
          if best_action is None: 
            print('\n\n\nBlocking Happend, executing default path\n\n\n')       
            if np.random.rand() < 0.5: 
                return motion_goals[0], Action.STAY
            else: 
                return self.get_lowest_cost_action_and_goal(start_pos_and_or, motion_goals)
        """  

        # 此时路被堵住了, 直接调用先前预处理好的 default 方案   
        if best_action is None: 
            print('\n\n\nBlocking Happend, executing default path\n\n\n')
            print('current position = {}'.format(start_pos_and_or)) 
            print('goal position = {}'.format(motion_goals))        
            if np.random.rand() < 0.5:  
                return None, Action.STAY
            else: 
                return self.get_lowest_cost_action_and_goal(start_pos_and_or, motion_goals)

        return best_goal, best_action


    def get_lowest_cost_action_and_goal(self, start_pos_and_or, motion_goals):
        """
        Chooses motion goal that has the lowest cost action plan.
        Returns the motion goal itself and the first action on the plan.
        """
        min_cost = np.Inf
        best_action, best_goal = None, None
        for goal in motion_goals:
            action_plan, _, plan_cost = self.mlam.motion_planner.get_plan(
                start_pos_and_or, goal
            )
            if plan_cost < min_cost:
                best_action = action_plan[0]
                min_cost = plan_cost
                best_goal = goal
        return best_goal, best_action

    def generate_success_feedback(self, state):
        success_feedback = f"Player {self.agent_index} succeeded at {self.current_ml_action}. "
        print(success_feedback)
        self.planner.add_msg_to_dialog_history({"role": "user", "content": success_feedback})

    def generate_failure_feedback(self, state):

        # if self.current_ml_action != 'wait':
        failure_feedback = self.generate_state_prompt(state)
        failure_feedback += f" Player {self.agent_index} failed at {self.current_ml_action}."
        # failure_feedback += " Why did  fail?"
        print(f"\n~~~~~~~~ Explainer~~~~~~~~\n{failure_feedback}")

        failure_message = {"role": "user", "content": failure_feedback}
        self.explainer.current_user_message = failure_message
        
        failure_explanation = self.explainer.query(self.openai_api_key())
        print(failure_explanation)

        self.explainer.add_msg_to_dialog_history({"role": "user", "content": failure_feedback})
        self.explainer.add_msg_to_dialog_history({"role": "assistant", "content": failure_explanation})

        self.planner.add_msg_to_dialog_history({"role": "user", "content": failure_explanation}) # 不加入错误解释
        

        # self.planner.dialog_history_list.pop() # delete the previous wrong skill query
        # self.planner.dialog_history_list.pop() # delete the previous state because it will generate again
        

        # return explanation

class GPTPlanningAgent(GPTAgent):
    def __init__(self, model="gpt-3.5-turbo-0301"):
        super().__init__(model=model)

