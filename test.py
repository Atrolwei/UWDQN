import time
from battle_field import BattleField
from agent import QLearningAgent, RandomAgent
from utils import to_xcoded_number, num2acts


def test_episode(env, agent):
    total_reward = 0
    obs = env.reset()
    env.render()
    while True:
        avail_actions=env.get_avail_actions()
        xcoded_obs=to_xcoded_number(obs,env.field_size)
        action = agent.predict(xcoded_obs,avail_actions)  # greedy
        if env.act_direction==5:
            action_real=[[2,4,5,6,8][idx] for idx in num2acts(action,env.N_preyers,5)]
        elif env.act_direction==9:
            action_real=[[1,2,3,4,5,6,7,8,9][idx] for idx in num2acts(action,env.N_preyers,9)]
        next_obs, reward, done, _ = env.step(action_real)
        total_reward += reward
        obs = next_obs
        time.sleep(0.5)
        env.render()
        if done:
            print('test reward = %.1f' % (total_reward))
            break
    return total_reward



def main():
    field_size=4
    N_preyers=3
    ele_goal=(1,2)
    episode_limit=15
    env = BattleField(field_size, N_preyers, ele_goal, episode_limit,act_direction=9)
    agent = QLearningAgent(
        obs_n=env.n_obss,
        act_n=env.n_actions,
        learning_rate=0.1,
        gamma=0.95,
        e_greed=0.2)
    agent.restore()
    # agent=RandomAgent(env.n_actions)

    # num_of_test
    num_of_test = 10
    total_reward = 0
    for _ in range(num_of_test):
        reward=test_episode(env, agent)
        total_reward += reward
    print('average reward = %.1f' % (total_reward / num_of_test))


if __name__ == "__main__":
    main()