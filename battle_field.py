import random

import numpy as np
from scipy.spatial import distance

from env_base import Env
from elephant import Elephant
from utils import num2dot,num2acts


class BattleField(Env):
    def __init__(self,
                 field_size,
                 N_preyers,
                 ele_goal,
                 episode_limit=15,
                 act_direction=5,
                 style='conservative'):
        super(BattleField,self).__init__()

        self.field_size=field_size
        self.N_preyers=N_preyers
        self.ele_goal=ele_goal        # deafault goal of the elephant, set a input variable if need
        self.episode_limit=episode_limit

        self.field_points=self.field_size**2
        self.n_obss=self.field_points**(N_preyers+1)
        self.act_direction=act_direction
        self.n_actions=self.act_direction**N_preyers

        self.ele_agent=Elephant(field_size,ele_goal,obs_range=2,style=style)      # the style can be choose from 'aggressive', 'moderate' and 'conservative'
        
        self.step_count=0
        self.last_obs=None
        self.last_action=None
        self.last_preyer_pos=None

    def sample_points(self):
        # sample (N_preyers+1) points from the field as a number list
        nums=random.sample(range(self.field_points),self.N_preyers+1)
        # transfer the number list to a point list using the function num2dot
        points=[num2dot(num,self.field_size) for num in nums]
        return points

    def reset(self):
        # randomly put the preyers and the elephant on the field
        points=self.sample_points()
        # if the elephant is at the goal or the preyers are at the same position, sample again
        while (points[0]==self.ele_goal or len(set(points))<self.N_preyers+1):
            points=self.sample_points()

        # assgin the initial position of the elephant and the preyers
        self.ele_pos=points[0]
        self.preyers_pos=points[1:]

        # reset the step count and the last observation and action
        self.step_count=0
        self.last_obs=points
        self.last_action=None
        self.last_preyer_pos=None
        return points    # return the xcoded number of the initial state


    def step(self,action):
        status=self._update_pos(action)
        obs=[self.ele_pos]+self.preyers_pos
        
        done,info=self.if_done(status)
        reward=self.calc_reward(done,info)

        self.step_count+=1
        self.last_obs=obs
        self.last_action=action
        self.last_preyer_pos=self.preyers_pos.copy()
        return obs,reward,done,info


    def _update_pos(self,action_preyers):
        status='normal'

        pos_set=set()
        pos_set.add(self.ele_pos)

        # the action of the preyers
        for idx,pos in enumerate(self.preyers_pos):
            act=action_preyers[idx]
            pos_new=self.get_x_y(pos,act)
            # check if the new position is valid
            if pos_new not in pos_set:
                pos_set.add(pos_new)
                self.preyers_pos[idx]=pos_new
            # if the new position is occupied by the elephant, the preyer stands still
            elif pos not in pos_set:
                pos_set.add(pos)
                self.preyers_pos[idx]=pos
            # if the new position is occupied by the other preyer, the preyer crashes
            else:
                status='crashed'
                print(f'ERROR! Preyer {idx+1} Crashes at {pos_new}!')
                continue
            
        if not status=='crashed':
            obs=[self.ele_pos]+self.preyers_pos
            # update the elephant's position
            act_ele=self.ele_agent.sample(obs)
            ele_pos_new=self.get_x_y(self.ele_pos,act_ele)
            self.ele_pos=ele_pos_new
        return status



    def if_done(self,status):
        done=False
        info_list=[]
        if self._if_surrounded():
            done=True
            info_list.append('Surrounded!')
        if self._if_time_out():
            done=True
            info_list.append('Time out!')
        if self._if_ele_wins():
            done=True
            info_list.append('Ele Wins!')
        if status=='crashed':
            done=True
            info_list.append('Crash!')
        if done:
            return True,info_list
        else:
            return False,[]


    def _if_surrounded(self):
        surrounded_list=[]
        for pos in self.preyers_pos:
            if distance.cityblock(self.ele_pos,pos)==1:
                surrounded_list.append(pos)
        N_preyers=self.N_preyers
        
        if not self._at_corner():               # when the elephant is not at the corner
            if len(surrounded_list)==N_preyers:
                edge_len_list=[]
                for idx in range(N_preyers):
                    if idx+1<N_preyers:
                        edge_len=np.linalg.norm(
                            np.array(surrounded_list[idx])-np.array(surrounded_list[idx+1]))
                        edge_len_list.append(edge_len)
                    else:
                        edge_len=np.linalg.norm(
                            np.array(surrounded_list[idx])-np.array(surrounded_list[0]))
                        edge_len_list.append(edge_len)
                max_len=np.array(edge_len_list).max()
                if abs(max_len-2)<1e-3:
                    return True
                else:
                    return False
            else:
                return False
        else:                   # the elephant at the corner
            if len(surrounded_list)==N_preyers-1:
                edge_len=np.linalg.norm(
                    np.array(surrounded_list[0])-np.array(surrounded_list[1]))
                if abs(edge_len-np.sqrt(2))<1e-3:
                    return True
                else:
                    return False
            else:
                return False

    def _at_corner(self):
        ele_pos=self.ele_pos
        field_size=self.field_size
        if ele_pos in [
            (0,0),
            (field_size-1,field_size-1),
            (0,field_size-1),
            (field_size-1,0)]:
            return True

    def _if_time_out(self):
        if self.step_count>self.episode_limit:
            return True
        else:
            return False

    def _if_ele_wins(self):
        if self.ele_pos==self.ele_goal:
            return True
        return False

    def get_x_y(self,pos,act):
        x,y=pos
        x_old,y_old=pos

        if act==1:
            y-=1
            x+=1
        elif act==2:
            x+=1
        elif act==3:
            y+=1
            x+=1
        elif act==4:
            y-=1
        elif act==5:
            pass
        elif act==6:
            y+=1
        elif act==7:
            y-=1
            x-=1
        elif act==8:
            x-=1
        elif act==9:
            y+=1
            x-=1
        else:
            pass

        size=self.field_size
        if x>size-1:
            x=size-1
            y=y_old
        elif x<0:
            x=0
            y=y_old
        if y>size-1:
            y=size-1
            x=x_old
        elif y<0:
            y=0
            x=x_old
        return x,y


    def render(self):
        field=np.zeros((self.field_size,self.field_size))
        field[self.ele_pos]=-1
        for idx,pos in enumerate(self.preyers_pos):
            field[pos]=idx+1
        print(field)
        

    def calc_reward(self,done,info):
        reward=0
        last_ele_pos=self.last_obs[0]   # the last position of the elephant is recoded in the last_obs
        ele_goal=self.ele_goal
        ele_pos=self.ele_pos
        vector0=np.array(last_ele_pos)-np.array(ele_goal)
        vector1=np.array(last_ele_pos)-np.array(ele_pos)
        num_vector=vector0.dot(vector1)
        if num_vector:
            reward=-num_vector/(np.linalg.norm(vector0)*np.linalg.norm(vector1))


        if self.last_preyer_pos:
            distance_old=0
            for pos in self.last_preyer_pos:
                distance_old+=np.linalg.norm(np.array(pos)-np.array(last_ele_pos))
            distance_new=0
            for pos in self.preyers_pos:
                distance_new+=np.linalg.norm(np.array(pos)-np.array(ele_pos))
            reward+=0.5*(distance_old-distance_new)


        if not done:
            return reward/20    # normalize the reward
        else:
            if 'Surrounded!' in info:
                reward+=40
            if 'Ele Wins!' in info:
                reward-=20
            if 'Crash!' in info:
                reward-=10
            
            return reward/20    # normalize the reward


    def get_avail_actions(self):
        avail_actions=np.ones(self.n_actions)
        for act_num in range(self.n_actions):
            acts=num2acts(act_num,self.N_preyers,xcimal=self.act_direction)
            if self.act_direction==5:
                acts=[[2,4,5,6,8][act] for act in acts]
            elif self.act_direction==9:
                acts=[[1,2,3,4,5,6,7,8,9][act] for act in acts]
            for preyer_idx,act in enumerate(acts):
                pos_new=self.get_x_y(self.preyers_pos[preyer_idx],act)
                if pos_new==self.preyers_pos[preyer_idx]:
                    avail_actions[act_num]=0
                    break
        return avail_actions      # return a one-hot form action
    

    
    
    