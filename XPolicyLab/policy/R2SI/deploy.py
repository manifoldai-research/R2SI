def _with_budget(task_env, observation, env_idx=0):
    """Copy only the official action count and native step limit into observation metadata.

    Nothing else of the environment is passed and nothing in the evaluation loop changes.
    """
    info = observation.get("additional_info")
    if not isinstance(info, dict):
        info = observation["additional_info"] = {}
    info.update(step=int(task_env.take_action_cnt[env_idx]), step_lim=int(task_env.step_lim))
    return observation


def _with_budget_batch(task_env, observations, env_indices):
    return [_with_budget(task_env, obs, env_idx) for obs, env_idx in zip(observations, env_indices)]


def eval_one_episode(TASK_ENV, model_client):

    model_client.call(func_name="reset") # reset policy

    while not TASK_ENV.is_episode_end(): # Check whether the episode ends
        obs = _with_budget(TASK_ENV, TASK_ENV.get_obs()) # Get Observation
        model_client.call(func_name="update_obs", obs=obs)  # Update Observation
        actions = model_client.call(func_name="get_action") # Get Action according to observation chunk

        for action_idx, action in enumerate(actions):
            TASK_ENV.take_action(action)

            if TASK_ENV.is_episode_end() or action_idx + 1 == len(actions):
                break

            obs = _with_budget(TASK_ENV, TASK_ENV.get_obs())
            model_client.call(func_name="update_obs", obs=obs)

def eval_one_episode_batch(TASK_ENV, model_client):

    model_client.call(func_name="reset")

    while not TASK_ENV.is_episode_end(): # Check whether the episode ends
        env_idx_list = TASK_ENV.get_running_env_idx_list() # Get Running Environment Index List
        obs_list = _with_budget_batch(TASK_ENV, TASK_ENV.get_obs_batch(env_idx_list), env_idx_list) # Get Observation
        model_client.call(func_name="update_obs_batch", obs=obs_list)
        actions = model_client.call(func_name="get_action_batch", obs=env_idx_list)  # Get Action according to observation chunk

        chunk_size = len(actions[0]) # Get the chunk size
        for action_idx in range(chunk_size): # Iterate over the action chunk
            current_action_list = [env_actions[action_idx] for env_actions in actions] # Get the current action list
            TASK_ENV.take_action_batch(current_action_list, env_idx_list) # Take the action

            if TASK_ENV.is_episode_end() or action_idx + 1 == chunk_size: # Check whether the episode ends
                break

            running = set(TASK_ENV.get_running_env_idx_list()) # Get the running environment index list
            active_batch_idx = [i for i, env_idx in enumerate(env_idx_list) if env_idx in running] # Get the active batch index

            actions = [actions[i] for i in active_batch_idx] # Get the active action list
            env_idx_list = [env_idx_list[i] for i in active_batch_idx] # Get the active environment index list
            model_client.call(func_name="update_obs_batch", obs=_with_budget_batch(TASK_ENV, TASK_ENV.get_obs_batch(env_idx_list), env_idx_list)) # Update the observation
