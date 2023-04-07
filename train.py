import time
from parl.utils import CSVLogger
from agent import QLearningAgent
from battle_field import BattleField
from utils import to_xcoded_number, num2acts


learn_logger = CSVLogger('./learn_log.csv')
def run_episode(env, agent, render=False):
    total_steps = 0  # 记录每个episode走了多少step
    total_reward = 0

    obs = env.reset()  # 重置环境, 重新开一局（即开始新的一个episode）

    while True:
        avail_actions=env.get_avail_actions()
        xcoded_obs=to_xcoded_number(obs,env.field_size)
        action = agent.sample(xcoded_obs,avail_actions)  # 根据算法选择一个动作
        if env.act_direction==5:
            action_real=[[2,4,5,6,8][idx] for idx in num2acts(action,env.N_preyers,5)]
        elif env.act_direction==9:
            action_real=[[1,2,3,4,5,6,7,8,9][idx] for idx in num2acts(action,env.N_preyers,9)]
        next_obs, reward, done, _ = env.step(action_real)  # 与环境进行一个交互

        # 训练 Q-learning算法
        xcoded_next_obs=to_xcoded_number(next_obs,env.field_size)
        learn_logger.log_dict({'rew':reward,'obs':obs,'next_obs':next_obs,'done':done})
        agent.learn(xcoded_obs, action, reward, xcoded_next_obs, done)

        obs = next_obs  # 存储上一个观察值
        total_reward += reward
        total_steps += 1  # 计算step数
        if render:
            env.render()  #渲染新的一帧图形
        if done:
            break
    return total_reward, total_steps


def test_episode(env, agent):
    total_reward = 0
    obs = env.reset()
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
            print('test reward = %.5f' % (total_reward))
            break


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
        e_greed=0.4)

    is_render = False
    for episode in range(10000):
        ep_reward, ep_steps = run_episode(env, agent, is_render)
        print('Episode %s: steps = %s , reward = %.5f' % (episode, ep_steps,
                                                          ep_reward))

        # 每隔20个episode渲染一下看看效果
        if episode % 20 == 0:
            is_render = True
        else:
            is_render = False
    # 训练结束，查看算法效果
    test_episode(env, agent)
    agent.save()


if __name__ == "__main__":
    main()
