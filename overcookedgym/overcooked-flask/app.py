import json,copy,argparse
import numpy as np
from flask import Flask, jsonify, request

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState, PlayerState, ObjectState
from overcooked_ai_py.planning.planners import MediumLevelPlanner
# from overcookedgym.overcooked_utils import NAME_TRANSLATION
from overcookedgym.GPTAgent import GPTMediumLevelAgent
from overcooked_ai_py.mdp.actions import Action

# from pantheonrl.common.trajsaver import SimultaneousTransitions
# from pantheonrl.tf_utils import get_pbt_agent_from_config
NAME_TRANSLATION = {
    "cramped_room": "simple",
    "asymmetric_advantages": "unident_s",
    "coordination_ring": "random1",
    "forced_coordination": "random0",
    "counter_circuit": "random3",
}

REVERSED_NAME_TRANSLATION = {v: k for k, v in NAME_TRANSLATION.items()}


app = Flask(__name__)
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--layout_name', type=str,
                    required=True, help="layout name")
parser.add_argument('--algo', type=int, default=1,
    help="0 or 1, 0 for MEP, 1 for your own algo")
parser.add_argument('--port', type=int,required=True,
                    help="port to run flask")
parser.add_argument('--seed', type=int,default=1,
                    help="seed for model")
parser.add_argument('--dummy', type=bool,default=False,
                    help="demo dummy partner. Won't move.")
global ARGS
ARGS = parser.parse_args()
def get_prediction(s, policy, layout_name, algo):
    if ARGS.dummy:
        return int(0)
    s=np.reshape(s,(1,s.shape[0],s.shape[1],s.shape[2]))
    action=policy(s)
    return int(action)


def process_state(state_dict, timestep):
    def object_from_dict(object_dict):
        return ObjectState(**object_dict)

    def player_from_dict(player_dict):
        held_obj = player_dict.get("held_object")
        if held_obj is not None:
            player_dict["held_object"] = object_from_dict(held_obj)
        return PlayerState(**player_dict)

    def state_from_dict(state_dict, timestep=timestep):
        state_dict["players"] = [player_from_dict(
            p) for p in state_dict["players"]]
        object_list = [object_from_dict(o)
                       for _, o in state_dict["objects"].items()]
        state_dict["objects"] = {ob.position: ob for ob in object_list}
        # state_dict['all_orders']=state_dict['order_list']
        # del state_dict['order_list']
        # del state_dict['all_orders']
        return OvercookedState(**state_dict, timestep=timestep)

    state = state_from_dict(copy.deepcopy(state_dict))
    # obs0,obs1=MDP.lossless_state_encoding(state)
    # return MDP.lossless_state_encoding(state)
    return state



def convert_traj_to_simultaneous_transitions(traj_dict, layout_name):

    ego_obs = []
    alt_obs = []
    ego_act = []
    alt_act = []
    flags = []

    for state_list in traj_dict['ep_states']:  # loop over episodes
        ego_obs.append([process_state(state, layout_name)[0]
                        for state in state_list])
        alt_obs.append([process_state(state, layout_name)[1]
                        for state in state_list])

        # check pantheonrl/common/wrappers.py for flag values
        flag = [0 for state in state_list]
        flag[-1] = 1
        flags.append(flag)

    for action_list in traj_dict['ep_actions']:  # loop over episodes
        ego_act.append([joint_action[0] for joint_action in action_list])
        alt_act.append([joint_action[1] for joint_action in action_list])

    ego_obs = np.concatenate(ego_obs, axis=-1)
    alt_obs = np.concatenate(alt_obs, axis=-1)
    ego_act = np.concatenate(ego_act, axis=-1)
    alt_act = np.concatenate(alt_act, axis=-1)
    flags = np.concatenate(flags, axis=-1)

    # return SimultaneousTransitions(
    return (
            ego_obs,
            ego_act,
            alt_obs,
            alt_act,
            flags,
        )



@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data_json = json.loads(request.data)
        state_dict, player_id_dict, server_layout_name, timestep = data_json["state"], data_json[
            "npc_index"], data_json["layout_name"], data_json["timestep"]
        player_id = int(player_id_dict)
        layout_name = NAME_TRANSLATION[server_layout_name]

        s = process_state(state_dict, timestep=timestep)

        print("---\n")

        # if player_id == 0:
        #     s, policy = s0, POLICY.direct_action
            # s, policy = s0, POLICY.action
        # elif player_id == 1:
            #s, policy = s1, POLICY_P1
            # s,policy=s1,POLICY.direct_action
            # s,policy=s1,POLICY.action
        # else:
        #     assert(False)
        algo=None
        # a = get_prediction(s, policy, layout_name, algo)
        action = POLICY.action(s)


        return jsonify({'action': Action.ACTION_TO_INDEX[action]})


@app.route('/updatemodel', methods=['POST'])
def updatemodel():
    '''if request.method == 'POST':
        data_json = json.loads(request.data)
        traj_dict, traj_id, server_layout_name, algo = data_json["traj"], data_json[
            "traj_id"], data_json["layout_name"], data_json["algo"]
        layout_name = NAME_TRANSLATION[server_layout_name]
        print(traj_id)

        if ARGS.trajs_savepath:
            # Save trajectory (save this to keep reward information)
            filename = "%s.json" % (ARGS.trajs_savepath)
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w') as f:
                json.dump(traj_dict, f)

            # Save transitions minimal (only state/action/done, no reward)
            simultaneous_transitions = convert_traj_to_simultaneous_transitions(
                traj_dict, layout_name)
            simultaneous_transitions.write_transition(ARGS.trajs_savepath)

        # Finetune model: todo

        REPLAY_TRAJ_IDX = 0
        done = True'''
        
    return jsonify({'status': True})


@app.route('/')
def root():
    if ARGS.dummy:
        return app.send_static_file('index_dummy.html')
    return app.send_static_file('index_'+str(ARGS.algo)+'.html')


if __name__ == '__main__':
    # if ARGS.algo==0:
    #     pbt_save_dir='./models/'+ARGS.layout_name+'/mep/'
    #     seed=ARGS.seed
    #     mep_agent=get_pbt_agent_from_config(pbt_save_dir, 30, seed=seed, agent_idx=0,iter=1)
    #     POLICY=mep_agent
    # else:
        #POLICY=your own agent
        # pass
    layout = ARGS.layout_name
    mdp = OvercookedGridworld.from_layout_name(layout)
    MLAM_PARAMS = {
        "start_orientations": False,
        "wait_allowed": True,
        "counter_goals": [],
        "counter_drop": [],
        "counter_pickup": [],
        "same_motion_goals": True,
    }
    counter_locations = mdp.get_counter_locations()
    MLAM_PARAMS["counter_goals"] = counter_locations
    MLAM_PARAMS["counter_drop"] = counter_locations
    MLAM_PARAMS["counter_pickup"] = counter_locations

    mlam = MediumLevelPlanner.from_pickle_or_compute(mdp, MLAM_PARAMS, force_compute=True).ml_action_manager
    agent = GPTMediumLevelAgent(mlam, REVERSED_NAME_TRANSLATION[layout])
    agent.set_mdp(mdp)
    agent.set_agent_index(1)
    POLICY = agent


    # TODO: client should pick layout name, instead of server?
    # currently both client/server pick layout name, and they must match
    MDP = OvercookedGridworld.from_layout_name(layout_name=ARGS.layout_name)

    app.run(debug=True, host='0.0.0.0',port=ARGS.port)
