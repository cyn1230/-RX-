def InferRules(init_obs, task):
    """
    Contains the rules for environment and task execute logic for different task types.
    Based on the Task ID 0-47 analysis:
    
    - Examination of a receptacle requires feedback indicating whether items are present.
    - Going to a receptacle should always be necessary before performing operations on it.
    """
    return """# Environment Rules

1. Before taking an object from a receptacle, you must examine the receptacle first.
2. You can only go to and interact with a receptacle after visiting it.
3. The environment will provide clear feedback about the presence or absence of objects when examined.

Remember to follow these rules to ensure task completion without violating the Environment's World Model."""
    

def WrapStep(env, init_obs, task, agent_action: str, logger):
    """
    Process the agent action and return the next observation, reward, and done status.
    This function ensures clear feedback is provided whenever an invalid action is attempted.
    """
    
    # Split actions based on semicolon for parsing complex multi-action commands
    actions = [a.strip() for a in agent_action.split(';') if a.strip()]
    
    for act in actions:
        obs, reward, done, info = env.step([act])
        
        if 'won' in info and not info['won']:
            # Handle invalid actions by providing clear feedback to the Agent
            observation = obs[0]
            logger.debug(f"Observation: {observation}")
            
            if "receptacle" in act.lower() and "examine" not in observation:
                return "\n".join(obs).replace("{'", '{"error_message": "You must examine the receptacle first before interacting with it."}'), None, False
            
            elif "go to" in act.lower() and "visited" not in observation:
                return "\n".join(obs).replace("{'", '{"error_message": "You need to visit the location before you can interact with it."}'), None, False
        
        # Update obs, reward, done based on the env step result
        obs, reward, done = obs[0], info['won'][0] if 'won' in info else False, done[0]
    
    return obs, reward, done