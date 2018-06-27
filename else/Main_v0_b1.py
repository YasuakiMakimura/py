# coding=utf-8

import os
from collections import defaultdict
from typing import Tuple, List, Union, ClassVar

import numpy as xp
# import cupy as xp
from pylab import *
import random

import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions


def min_max(val: object, min_val: object, max_val: object) -> object:
    if val < min_val:  # ある値が下限以下である場合
        return min_val
    elif val > max_val:  # ある値が上限以上である場合
        return max_val
    return val  # ある値がちゃんと指定した値域内に収まる場合はそのままある値を返す


# noinspection PyMissingOrEmptyDocstring,PyBroadException
class Database:
    def __init__(self, nm_dir_datalist: str = 'data', b_save: bool = True):
        self.p = defaultdict(lambda: 'none')
        os.makedirs(nm_dir_datalist, exist_ok=True)
        files = os.listdir(f'{nm_dir_datalist}/')
        print(f'List of "folder" in {nm_dir_datalist}/: ')
        for file in files:
            print(file)
        try:
            self.nm_dir_data = input(f"\n Name of data folder: ")
            os.makedirs(f'{nm_dir_datalist}/{self.nm_dir_data}')
        except Exception as error:
            print(f'{type(error)}, Please appoint name of data folder, again')
            self.nm_dir_data = input(f"Name of data folder: ")
            os.makedirs(f'{nm_dir_datalist}/{self.nm_dir_data}')
        finally:
            self.p['path_dir_data'] = f'{nm_dir_datalist}/{self.nm_dir_data}/'
        self.p['path_dir_traj'] = f'{self.p["path_dir_data"]}traj/'
        os.makedirs(self.p['path_dir_traj'], exist_ok=True)
        self.p['path_dir_ro'] = f'{self.p["path_dir_data"]}ro/'
        os.makedirs(self.p['path_dir_ro'], exist_ok=True)
        self.p['path_dir_lc'] = f'{self.p["path_dir_data"]}lc/'
        os.makedirs(self.p['path_dir_lc'], exist_ok=True)
        self.p['path_dir_critic'] = f'{self.p["path_dir_data"]}critic/'
        os.makedirs(self.p['path_dir_critic'], exist_ok=True)
        self.p['path_dir_exp'] = f'{self.p["path_dir_data"]}exp_component/'
        os.makedirs(self.p['path_dir_exp'], exist_ok=True)
        self.p['path_dir_c_v'] = f'{self.p["path_dir_data"]}c_v/'
        os.makedirs(self.p['path_dir_c_v'], exist_ok=True)
        self.p['path_dir_td'] = f'{self.p["path_dir_data"]}td/'
        os.makedirs(self.p['path_dir_td'], exist_ok=True)
        self.p['path_dir_rnd'] = f'{self.p["path_dir_data"]}rnd_for_a_episode/'
        os.makedirs(self.p['path_dir_rnd'], exist_ok=True)
        self.p['path_dir_actor'] = f'{self.p["path_dir_data"]}actor/'
        os.makedirs(self.p['path_dir_actor'], exist_ok=True)
        self.p['path_dir_sw_in'] = f'{self.p["path_dir_data"]}sw_in/'
        os.makedirs(self.p['path_dir_sw_in'], exist_ok=True)

        self.p['pl_cycle'] = 1000
        # self.p['test_cycle'] = 100
        self.p['n_pl_stau'] = 10
        self.p['n_pl_ltau'] = 10
        self.p['n_ep'] = 2000
        # self.p['n_ep_test'] = 0
        self.p['max_step'] = 200
        self.p['pl_cycle_ave_step'] = 100
        self.p['cycle_save_model'] = 1000
        self.p['n_in'] = 7
        self.p['n_h_neuron'] = 1000
        self.p['n_fb'] = 10
        self.p['p_connect'] = 10
        self.p['v_lambda'] = 1.5
        self.p['v_tau']: Tuple[Union[int, float]] = (10, 2)
        for off_v_tau in self.p['v_tau']:
            self.p[f'path_dir_tau{off_v_tau}'] = f'{self.p["path_dir_ro"]}tau{off_v_tau}/'
            os.makedirs(self.p[f'path_dir_tau{off_v_tau}'], exist_ok=True)
        self.p['path_dir_fb'] = f'{self.p["path_dir_ro"]}fb/'
        os.makedirs(self.p['path_dir_fb'], exist_ok=True)

        self.p['n_tau']: Tuple[int] = (700, 300)
        self.p['net_in_wscale'] = 1.0
        self.p['net_fb_wscale'] = 1.0
        self.p['net_type'] = 1

        self.p['out_l1'] = 100
        self.p['out_l2'] = 40
        self.p['out_l3'] = 10
        self.p['out_top'] = 3
        self.p['mlr_nobias'] = (False, False, False, False, False)  # bypass, l1, 12, 13, top(critic, actor)

        self.p['field_x'] = 10.0
        self.p['field_y'] = 10.0

        self.p['seed'] = 1
        self.p['init_rnd_scale'] = 1.0
        self.p['finish_rnd_scale'] = 1.0
        self.p['learning_rate'] = 0.01
        self.p['alpha'] = 0.1
        if self.p['alpha'] >= 1.0:
            raise ValueError('Not smoothing_coefficient >= 1.0')
        self.p['reward'] = 0.8
        self.p['wall_penalty'] = -0.0
        self.p['pena_without_switching'] = -0.8
        self.p['discount_rate'] = 0.9
        self.p['t_actor_limit'] = 0.8
        self.p['sw_scale'] = 1.0  # todo 日付保存,csvファイルで保存または画像をパラメータと一緒に画像として保存、メモもできるようにする

        self.p['add_exploration'] = True
        self.p['moving_ave'] = True
        if not self.p['moving_ave']:
            self.p['add_exploration'] = True

        self.p['load_w'] = (False, f"{nm_dir_datalist}/"
                                  f"{'test2018_1_30_2_28_seed1'}/"
                                  f"{'mrl_ep99999.npz'}")
        if self.p['load_w'][0]:
            judge = input(f'Directory of load file: {self.p["load_w"]}, ok? y or n: ')
            if judge == 'n':
                sys.exit(1)
            elif judge == 'y':
                pass
            else:
                raise Exception('Please input "y" or "n"')

        flag_uniform_out = False
        flag_lecunnormal_out = False
        flag_uniform_bypass = False
        if flag_uniform_out:
            self.p['out_uni_ws'] = (True, (0.5, 0.5, 0.5, 0.5))  # l1, 12, 13, top
        else:
            self.p['out_uni_ws'] = (False,)  # ws: weight scale

        if flag_lecunnormal_out:
            self.p['out_lecnor_ws'] = (True, (4.0, 4.0, 4.0, 4.0))  # l1, 12, 13, top
        else:
            self.p['out_lecnor_ws'] = (False,)  # ws: weight scale

        if flag_uniform_bypass:
            self.p['bypass_ws'] = (True, 1.0)
        else:
            self.p['bypass_ws'] = (False, None)

        if b_save:
            daytime = datetime.datetime.today()
            self.f = open(f'{self.p["path_dir_data"]}para', "a")
            self.f.write(str(daytime) + "\n")
            for k, v in self.p.items():
                self.f.write(k + "=" + str(v) + "\n")
            self.f.close()


if __name__ == "__main__":
    # rc, pc, sc = [0] * 3

    def plot_in_steploop():
        tsk.add_p_log2(posi_x=tsk.x_agent, posi_y=tsk.y_agent)
        for sss in range(db.p['n_pl_stau']):
            list_stau[sss].append(net.ro[sss])
        for lll in range(-1, -db.p['n_pl_ltau'] - 1, -1):
            list_ltau[lll].append(net.ro[lll])
        for fff in range(db.p['out_l3']):
            list_fb[fff].append(out_o['3rd'].data[0][fff])
        list_critic.append(out_o['top'].data[0][0])


    def plot_after_steploop():
        tsk.replot2()
        tsk.replot_goal()
        tsk.save(fig_name=f'{db.p["path_dir_traj"]}ep{ep}_traj')
        stau.subplot(list_stau)
        ltau.subplot(list_ltau)
        fb.subplot(list_fb)
        stau.save(nm_fig=db.p[f'path_dir_tau{stau.k["inf"]}'] + f'ep{ep}_tau{stau.k["inf"]}')
        ltau.save(nm_fig=db.p[f'path_dir_tau{ltau.k["inf"]}'] + f'ep{ep}_tau{ltau.k["inf"]}')
        fb.save(nm_fig=f'{db.p["path_dir_fb"]}fb_ep{ep}')
        critic.plot(list_data=list_critic)
        critic.save(nm_fig=f'{db.p["path_dir_critic"]}critic_ep{ep}')
        # tde.plot(list_data=list_td, x_list=range(len(list_td)))
        tde.plot(list_data=list_td)
        tde.save(nm_fig=f'{db.p["path_dir_td"]}td_{ep}')
        # rnd.multi_plot(lists_data=(list_rndx, list_rndy), x_lists=(range(len(list_rndx)), range(len(list_rndy))))
        rnd.multi_plot(lists_data=(list_rndx, list_rndy))
        rnd.save(nm_fig=f'{db.p["path_dir_rnd"]}rnd_{ep}')


    def plot_after_eploop():
        # lc.multi_plot(lists_data=(list_finish_step, list_ave_step), x_lists=(range(len(list_finish_step)), range(db.p['pl_cycle_ave_step'] - 1, len(list_finish_step), db.p['pl_cycle_ave_step'])))
        lc.multi_plot(lists_data=(list_ave_step,), x_lists=(range(db.p['pl_cycle_ave_step'] - 1, len(list_finish_step), db.p['pl_cycle_ave_step']),))
        lc.save(nm_fig=f'{db.p["path_dir_lc"]}learning_curve')  # c_v.plot(list_data=list_c_v)  # c_v.save(nm_fig=f'{db.p["path_dir_c_v"]}c_v')

        exp.multi_plot(lists_data=(list_ave_exp_x, list_ave_exp_y), x_lists=(range(db.p['n_ep']), range(db.p['n_ep'])))
        exp.save(nm_fig=f'{db.p["path_dir_exp"]}exp')
        # ave_exp.multi_plot(lists_data=(list_aveave_exp_x, list_aveave_exp_y), x_lists=(range(db.p['pl_cycle_ave_step'] - 1, len(list_ave_exp_x), db.p['pl_cycle_ave_step']), range(db.p['pl_cycle_ave_step'] - 1, len(list_ave_exp_y), db.p['pl_cycle_ave_step'])))  # ave_exp.save(nm_fig=f'{db.p["path_dir_exp"]}ave_exp')
        # print(f'rc:{rc}')
        # print(f'pc:{pc}')
        # print(f'sc:{sc}')
        # pie.out([rc, pc, sc])

    # def reward_count(reward):
    #     global rc
    #     if reward == db.p['reward']:
    #         rc += 1
    #
    #
    # def penalty_count(penalty):
    #     global pc
    #     if penalty == db.p['pena_without_switching']:
    #         pc += 1
    #
    # # def rp_count(reward):
    # #     reward_count(reward)
    # #     penalty_count(penalty=reward)
    #
    # def limitstep_count(step):
    #     global sc
    #     max_step = db.p['max_step']
    #     max_step -= 1
    #     if step == max_step:
    #         sc += 1


    db = Database()  # todo saveを最後でやる？　途中でパラメータを変えれるようにするため？
    import Field_goal as fd
    import Neu_Reservoir_b2 as neu
    # import Neu_Reservoir as neu
    import Net_chainer_ReNN_b4 as mlr
    import Logger as log
    # import Graphic_pro,Pie_v0 as cir

    # import Analysis as ana

    random.seed(db.p['seed'])
    np.random.seed(db.p['seed'])
    xp.random.seed(db.p['seed'])

    # pie = cir.Pie('count')
    # pie.label = ['reward', 'penalty', 'limit step']
    net = neu.Reservoir(db.p['n_in'], db.p['n_h_neuron'], db.p['n_fb'], db.p['v_tau'], db.p['n_tau'], p_connect=db.p['p_connect'], v_lambda=db.p['v_lambda'], in_wscale=db.p['net_in_wscale'], fb_wscale=db.p['net_fb_wscale'])
    out = mlr.MyChain(n_in=db.p['n_h_neuron'], n_bypass=db.p['n_in'], n_l1=db.p['out_l1'], n_l2=db.p['out_l2'], n_l3=db.p['out_l3'], n_top=db.p['out_top'], nor_w=db.p['out_lecnor_ws'], uni_w=db.p['out_uni_ws'],
                      uni_bypass_w=db.p['bypass_ws'], nobias=db.p['mlr_nobias'])
    optimizer = optimizers.SGD(lr=db.p["learning_rate"])
    # noinspection PyDeprecation
    optimizer.use_cleargrads()
    optimizer.setup(out)
    if db.p['load_w'][0]:
        serializers.load_npz(db.p['load_w'][1], out)
    tsk = fd.Field(field_x=db.p['field_x'], field_y=db.p['field_y'], Num_plot=1, fig_number=0, rew=db.p['reward'], pena=db.p['wall_penalty'], spena=db.p['pena_without_switching'], sw_scale=db.p['sw_scale'])
    # lc = log.Log(n_line=2, num_fig=1, c=('red', 'blue'), xlabel='episode', ylabel='step', y_high=db.p['max_step'] + 25, x_high=db.p['n_ep'], auto_xrange=False)
    lc = log.Log(n_line=1, num_fig=1, c='blue', xlabel='episode', ylabel='step', y_high=db.p['max_step'] + 25, x_high=db.p['n_ep'], auto_xrange=False, grid=True)
    # test_lc = log.Log(n_line=2, num_fig=2, c=('red', 'blue'), xlabel='episode', ylabel='step', y_high=db.p['max_step'] + 25, x_high=db.p['n_ep_test'], auto_xrange=False)
    critic = log.Log(n_line=1, num_fig=3, y_low=-1.1, y_high=1.1, x_high=100, xlabel='step', ylabel='critic', c='magenta', origin_line=True, any_line=True, iter_any=[db.p['reward']] * 100)
    stau = log.Log(n_line=db.p['n_pl_stau'], num_fig=4, subplot=True, y_low=-1.1, y_high=1.1, figsize=(8, 6), x_high=db.p['max_step'], inf=db.p['v_tau'][1], c='red', xlabel='episode', ylabel='output(τ: 2)')
    ltau = log.Log(n_line=db.p['n_pl_ltau'], num_fig=5, subplot=True, y_low=-1.1, y_high=1.1, figsize=(8, 6), x_high=db.p['max_step'], inf=db.p['v_tau'][0], xlabel='episode', ylabel='output(τ: 10)')
    fb = log.Log(n_line=db.p['out_l3'], num_fig=6, subplot=True, y_low=-1.1, y_high=1.1, figsize=(8, 6), x_high=db.p['max_step'], c='green', xlabel='episode', ylabel='output(feedback)')
    exp = log.Log(n_line=2, num_fig=7, y_low=0, y_high=1.1, x_high=db.p['n_ep'], c=('#000080', '#FF4500'), xlabel='episode', ylabel='exploration component', auto_xrange=False, legend=('Actor x', 'Actor y'), alpha=(0.5, 0.5), grid=True)
    tde = log.Log(n_line=1, num_fig=8, y_low=-1.1, y_high=1.1, x_high=db.p['max_step'], c='#8A2BE2', xlabel='step', ylabel='TD error', grid=True)
    rnd = log.Log(n_line=2, num_fig=9, y_low=-1.1, y_high=1.1, x_high=db.p['max_step'], c=('#A52A2A', '#5F9EA0'), xlabel='step', ylabel='exploration component', legend=('exp of actor x', 'exp of actor y'), alpha=(0.5, 0.5), grid=True)


    # ave_exp = log.Log(n_line=2, num_fig=8, y_low=-1.5, y_high=1.5, x_high=db.p['n_ep'], c=('#FF6347', '#7B68EE'), xlabel='episode', ylabel='exploration component', auto_xrange=False, legend=('Actor x', 'Actor y'), alpha=(0.5, 0.5))
    # c_v = log.Log(n_line=1, num_fig=9, y_low=0, y_high=1.1, x_high=db.p['n_ep'], xlabel='episode', ylabel='Coefficient of Variation', c='#008080')

    # train = Train(database=db, tsk=tsk, net=net, out=out, external_rnd=db.p['add_exploration'])
    list_finish_step = []
    list_ave_step = []
    # list_c_v = []
    list_exp_x, list_ave_exp_x, list_aveave_exp_x = [], [], []
    list_exp_y, list_ave_exp_y, list_aveave_exp_y = [], [], []
    list_td, list_rndx, list_rndy = [], [], []

    out_o = {}
    i = {}
    pre_i = {}
    init_rnd_scale = db.p['init_rnd_scale']
    for ep in range(db.p['n_ep']):
        print(f'ep:{ep}')
        if ep != 0:
            net.reset_net()
        if (ep % db.p['cycle_save_model'] == 0) or (ep == db.p['n_ep'] - 1):
            serializers.save_npz(f'{db.p["path_dir_data"]}mrl_ep{ep}.npz', out)
        tsk.all_p_random_set()
        i['sw_in'], i['d_g'], i['d_sw'], i['sin_g'], i['cos_g'], i['sin_sw'], i['cos_sw'] = tsk.get_state()
        in_inf: Tuple[float, ...] = tuple(i.values())
        net.in_o[:] = in_inf  # type: np.ndarray
        bypass = Variable(xp.array([in_inf], dtype=np.float32))
        net.ru_comp()
        net.ro_comp()
        out_in = Variable(xp.array([net.ro], dtype=np.float32))
        if (ep % db.p['pl_cycle'] == 0) or (ep == db.p['n_ep'] - 1):
            tsk.ini_add_p_log(posi_x=tsk.x_agent, posi_y=tsk.y_agent)
            list_stau = [[] for j in range(db.p['n_pl_stau'])]
            list_ltau = [[] for j in range(db.p['n_pl_ltau'])]
            list_fb = [[] for j in range(db.p['out_l3'])]
            list_critic = []
        # list_d = []

        for step in range(db.p['max_step']):
            out_o['1st'], out_o['2nd'], out_o['3rd'], out_o['top'] = out.ff_comp(net_in=out_in, bypass=bypass, net_type=db.p['net_type'])
            out2 = out.copy()
            net.read_o = out_o['3rd'].data[0]

            if db.p['moving_ave']:
                if step == 0:
                    mma_x, mma_y = db.p['alpha'] * out_o['top'].data[0][1], db.p['alpha'] * out_o['top'].data[0][2]
                else:
                    old_mma_x, old_mma_y = mma_x, mma_y
                    mma_x = (1 - db.p['alpha']) * old_mma_x + db.p['alpha'] * out_o['top'].data[0][1]
                    mma_y = (1 - db.p['alpha']) * old_mma_y + db.p['alpha'] * out_o['top'].data[0][2]
                rndx, rndy = out_o['top'].data[0][1] - mma_x, out_o['top'].data[0][2] - mma_y
                if db.p['add_exploration']:
                    tsk.move_agent(out_o['top'].data[0][1] + rndx, out_o['top'].data[0][2] + rndy)  # list_exp_x.append(rndx * 2)  # list_exp_y.append(rndy * 2)
                else:
                    tsk.move_agent(out_o['top'].data[0][1], out_o['top'].data[0][2])  # list_exp_x.append(rndx)  # list_exp_y.append(rndy)

            else:
                # rndx, rndy = random.uniform(-init_rnd_scale, init_rnd_scale), random.uniform(-init_rnd_scale, init_rnd_scale)
                rndx, rndy = np.random.uniform(-init_rnd_scale, init_rnd_scale, (2,))
                tsk.move_agent(out_o['top'].data[0][1] + rndx, out_o['top'].data[0][2] + rndy)  # list_exp_x.append(rndx + (out_o['top'].data[0][1] - mma_x))  # list_exp_y.append(rndy + (out_o['top'].data[0][2] - mma_y))
            # list_d.append(math.sqrt((tsk.x_agent - db.p['field_x']/2)**2 + (tsk.y_agent - db.p['field_y']/2)**2))
            list_exp_x.append(abs(rndx))
            list_exp_y.append(abs(rndy))

            tsk.agent_crash_wall()
            reward, str_state = tsk.state_check()  # type: float, str
            if (ep % db.p['pl_cycle'] == 0) or (ep == db.p['n_ep'] - 1):
                plot_in_steploop()
            i['sw_in'], i['d_g'], i['d_sw'], i['sin_g'], i['cos_g'], i['sin_sw'], i['cos_sw'] = tsk.get_state()
            in_inf: Tuple[float, ...] = tuple(i.values())
            bypass = Variable(xp.array([in_inf], dtype=np.float32))
            net.in_o[:] = in_inf  # type: np.ndarray
            net.ru_comp()
            net.ro_comp()
            out_in = Variable(xp.array([net.ro], dtype=np.float32))
            out_o['pre_1st'], out_o['pre_2nd'], out_o['pre_3rd'], out_o['pre_top'] = out2.ff_comp(net_in=out_in, bypass=bypass, net_type=db.p['net_type'])
            if str_state in {'goal', 'out'}:
                out_o['pre_top'] = Variable(xp.array([[0.0]], dtype=np.float32))
            td = reward + db.p['discount_rate'] * out_o['pre_top'].data[0][0] - out_o['top'].data[0][0]
            # reward_count(reward)
            # penalty_count(reward)
            t_critic = out_o['top'].data[0][0] + td
            t_actorx = out_o['top'].data[0][1] + rndx * td
            t_actory = out_o['top'].data[0][2] + rndy * td
            list_td.append(td)
            list_rndx.append(rndx)
            list_rndy.append(rndy)

            teach = Variable(xp.array([[t_critic, min_max(t_actorx, -db.p['t_actor_limit'], db.p['t_actor_limit']), min_max(t_actory, - db.p['t_actor_limit'], db.p['t_actor_limit'])]], dtype=np.float32))  # todo ここはxpじゃなくてnpで良いのか
            out.cleargrads()
            loss = F.mean_squared_error(out_o['top'], teach)
            loss.backward()
            optimizer.update()  # todo stepのloop外の処理も関数としてかく？ todo saveの関数を作る？
            if str_state in {'goal', 'out'}:
                if str_state == 'out':
                    print(f'step(out): {step}')
                    step = 199
                break
            """STEP_END"""
        list_finish_step.append(step)
        # limitstep_count(step)
        # list_c_v.append(ana.c_v(list_d))

        print(f'step: {step}, str_state: {str_state}, rnd_scale: {init_rnd_scale}\n')
        if (db.p['init_rnd_scale'] > db.p['finish_rnd_scale']) and (not db.p['moving_ave']):
            init_rnd_scale = init_rnd_scale * pow(db.p['finish_rnd_scale'] / db.p['init_rnd_scale'], 1.0 / db.p['n_ep'])
        if (ep % db.p['pl_cycle'] == 0) or (ep == db.p['n_ep'] - 1):
            plot_after_steploop()
        # noinspection PyUnboundLocalVariable
        list_td, list_rndx, list_rndy = [], [], []
        list_ave_exp_x.append(sum(list_exp_x[0: len(list_exp_x)]) / len(list_exp_x))
        list_ave_exp_y.append(sum(list_exp_y[0: len(list_exp_y)]) / len(list_exp_y))
        list_exp_x, list_exp_y = [], []
        if len(list_finish_step) % db.p['pl_cycle_ave_step'] == 0:
            list_ave_step.append(sum(list_finish_step[len(list_finish_step) - db.p['pl_cycle_ave_step']: len(list_finish_step)]) / db.p[
                'pl_cycle_ave_step'])  # list_aveave_exp_x.append(sum(list_ave_exp_x[len(list_ave_exp_x) - db.p['pl_cycle_ave_step']: len(list_ave_exp_x)]) / db.p['pl_cycle_ave_step'])  # list_aveave_exp_y.append(sum(list_ave_exp_y[len(list_ave_exp_y) - db.p['pl_cycle_ave_step']: len(list_ave_exp_y)]) / db.p['pl_cycle_ave_step'])
        # if len(list_exp_x) % db.p['pl_cycle_ave_step'] == 0:

        """EPISODE_END"""
    plot_after_eploop()
