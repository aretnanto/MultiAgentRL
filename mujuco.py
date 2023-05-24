from shimmy import MeltingPotCompatibilityV0
env = MeltingPotCompatibilityV0(substrate_name="prisoners_dilemma_in_the_matrix__arena", render_mode="human")
observations = env.reset()
while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    env.step(actions)
env.close()