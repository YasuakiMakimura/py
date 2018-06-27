# coding=utf-8
import numpy as np
# import cupy as xp
import chainer.functions as F
from chainer import Variable
from chainer import optimizers, serializers
import Directory_Processor
import Field_goal
import Past_Reservoir
import Layered_NN
import Bokeh_line


class Database:
    para = {'learning_type': "conventional", 'seed': 1, 'learning_rate': 0.01, 'episode': 100,
            'limit_step': 200, 'plot_cycle': 1, 'model_savecycle': 1000,
            'n_neuron': {'in': 7, 'reservoir': 1000, 'fb': 10}, 'connect_rate': 10,
            'wscale': {'in': 0.0, 'fb': 1.0},
            'n_neuron_of_readout': {'l1': 100, 'l2': 40, 'l3': 10, 'top': 3}, 'lambda': 1.5,
            'reward': 0.8, 'penalty_for_wallcrash': -0.1, 'penalty_for_non-switching': -0.8,
            'switch_flag': 10, 'discount_rate': 0.9, 'smoothing_factor': 0.1,
            'load_model': False,
            'load_file':
                '/home/maki/MEGA/tex/reservoir/RL/2018/6_22/Figure/0622_0757_02/mlr_model/ep49999'
                '.npz'}

    def __init__(self):
        self.dirs = Directory_Processor.DirectoryProcessing()
        self.dirs.root_name = 'data'
        with open(f'{self.dirs.data_folders["subroot"]}para', 'w') as f:
            for key_name, value in self.para.items():
                f.write(f'{key_name} = {str(value)}"\n"')


def reservoir_reset(*reservoir_objects):
    for reservoir in reservoir_objects:
        reservoir.reset_net()


def mlr_save(model, file_name):
    return serializers.save_npz(file_name, model)


def initial_pos_set(task_object, set_type='random'):
    if set_type == 'random':
        return task_object.all_p_random_set()


def ro_comp(input_signal, *reservoir_objects):
    for reservoir in reservoir_objects:
        reservoir.in_o[:] = input_signal
        reservoir.ru_comp()
        reservoir.ro_comp()
        yield reservoir.ro


def feedback_to_reservoir_comp(feedback, *reservoir_objects):
    for reservoir in reservoir_objects:
        reservoir.read_o = feedback


def to_variable(data):
    return Variable(np.array([data], dtype=np.float32))


def moving_average_comp(smoothing_factor, past_moving_average, current_input):
    return (1 - smoothing_factor) * past_moving_average + smoothing_factor * current_input


def td_error_comp(discount_rate, reward, critic, pre_critic):
    return reward + discount_rate * pre_critic - critic


def teach_signal_comp(critic, actorx, actory, td_error, exp):
    teach_critic = critic + td_error
    teach_actorx = actorx + exp[0] * td_error
    teach_actory = actory + exp[1] * td_error
    return teach_critic, teach_actorx, teach_actory


def min_max(val: object, min_val: object, max_val: object) -> object:
    if val < min_val:  # ある値が下限以下である場合
        return min_val
    elif val > max_val:  # ある値が上限以上である場合
        return max_val
    return val  # ある値がちゃんと指定した値域内に収まる場合はそのままある値を返す


def main():
    database = Database()
    database.dirs.adding_datafolder('trajectory', 'mlr_model', 'ro_tau2', 'ro_tau10')
    para = database.para
    with open(f'{database.dirs.data_folders["subroot"]}multi_reservoir_type', 'w') as f:
        f.write(f'')
    with open(f'{database.dirs.data_folders["subroot"]}{para["learning_type"]}', 'w') as f1:
        f1.write(f'')
    np.random.seed(para['seed'])

    task = Field_goal.Field(Num_plot=1, fig_number=0, rew=para['reward'],
                            pena=para['penalty_for_wallcrash'],
                            spena=para['penalty_for_non-switching'], sw_scale=para['switch_flag'])

    reservoir_tau2 = Past_Reservoir.Reservoir(para['n_neuron']['in'], 300,
                                              para['n_neuron']['fb'], v_tau=(2,), n_tau=(300, ),
                                              p_connect=para['connect_rate'],
                                              v_lambda=para['lambda'],
                                              in_wscale=para['wscale']['in'],
                                              fb_wscale=para['wscale']['fb'])

    reservoir_tau10 = Past_Reservoir.Reservoir(para['n_neuron']['in'],
                                               700,
                                               para['n_neuron']['fb'], v_tau=(10,), n_tau=(700,),
                                               p_connect=para['connect_rate'],
                                               v_lambda=para['lambda'],
                                               in_wscale=para['wscale']['in'],
                                               fb_wscale=para['wscale']['fb'])

    mlr = Layered_NN.MyChain(para['n_neuron']['reservoir'] * 1, para['n_neuron']['in'],
                             para['n_neuron_of_readout']['l1'], para['n_neuron_of_readout']['l2'],
                             para['n_neuron_of_readout']['l3'], para['n_neuron_of_readout']['top'])

    optimizer = optimizers.SGD(lr=para['learning_rate'])
    optimizer.use_cleargrads()
    optimizer.setup(mlr)
    """LOAD MODEL"""
    if para['load_model']:
        serializers.load_npz(para['load_file'], mlr)
        input('model load, ok? Please Enter')
    """GRAPHICAL INSTANCE"""
    learning_curve = Bokeh_line.Line(n_line=2, xlabel='episode', ylabel='step')
    reservoir_tau2_output = Bokeh_line.Line(n_line=10, xlabel='step', ylabel='output(τ=2)')
    reservoir_tau10_output = Bokeh_line.Line(n_line=10, xlabel='step', ylabel='output(τ=10)')
    input_series = Bokeh_line.Line(n_line=para['n_neuron']['in'], xlabel='step', ylabel='input')

    ma_x, ma_y = 0, 0
    """EPISODELOOP_START"""
    for epi in range(para['episode']):
        print(f'epi: {epi}')
        if epi > 0:
            reservoir_reset(reservoir_tau2, reservoir_tau10)
        if epi % para['model_savecycle'] == 0 or epi == para['episode'] - 1:
            mlr_save(mlr, f'{database.dirs.data_folders["mlr_model"]}ep{epi}.npz')
        initial_pos_set(task, set_type='random')
        input_signal = task.get_state()
        bypass = to_variable(input_signal)
        input_to_mlr = to_variable(
            np.hstack(ro_comp(input_signal, reservoir_tau2, reservoir_tau10)))
        if (epi % para['plot_cycle'] == 0) or (epi == para['episode'] - 1):
            task.ini_add_p_log(posi_x=task.x_agent, posi_y=task.y_agent)
        """STEPLOOP_START"""
        for step in range(para['limit_step']):
            mlr_output = list(mlr.ff_comp(net_in=input_to_mlr, bypass=bypass))
            feedback_to_reservoir_comp(mlr_output[-2].data[0], reservoir_tau2, reservoir_tau10)
            if para['learning_type'] in ['non-emphasized', 'emphasized']:
                ma_x = moving_average_comp(para['smoothing_factor'], ma_x,
                                           mlr_output[-1].data[0][1])
                ma_y = moving_average_comp(para['smoothing_factor'], ma_y,
                                           mlr_output[-1].data[0][2])
                exp_x = mlr_output[-1].data[0][1] - ma_x
                exp_y = mlr_output[-1].data[0][2] - ma_y
            elif para['learning_type'] == 'conventional':
                exp_x, exp_y = np.random.uniform(-1, 1, (2,))
            else:
                raise Exception(f'Not expected error??????')
            if para['learning_type'] in ['emphasized', 'conventional']:
                task.move_agent(mlr_output[-1].data[0][1] + exp_x,
                                mlr_output[-1].data[0][2] + exp_y)
            elif para['learning_type'] == 'non-emphasized':
                task.move_agent(mlr_output[-1].data[0][1], mlr_output[-1].data[0][2])
            else:
                raise Exception(f'Not expected error!!!!')
            task.agent_crash_wall()
            reward, state = task.state_check()
            """DATASTORE"""
            if (epi % para['plot_cycle'] == 0) or (epi == para['episode'] - 1):
                task.add_p_log2(posi_x=task.x_agent, posi_y=task.y_agent)
                for offset, neuron in enumerate(reservoir_tau2.ro[0:10]):
                    reservoir_tau2_output.store(neuron, offset=offset)
                for offset, neuron in enumerate(reservoir_tau10.ro[0:10]):
                    reservoir_tau10_output.store(neuron, offset=offset)
            input_signal = task.get_state()
            bypass = to_variable(input_signal)
            input_to_mlr = to_variable(
                    np.hstack(ro_comp(input_signal, reservoir_tau2, reservoir_tau10)))
            mlr_pre_output = list(mlr.ff_comp(net_in=input_to_mlr, bypass=bypass))
            if state in {'goal', 'out'}:
                mlr_pre_output[-1] = to_variable([0.0])
            td_error = td_error_comp(para['discount_rate'], reward, mlr_output[-1].data[0][0],
                                     mlr_pre_output[-1].data[0][0])

            teach = teach_signal_comp(mlr_output[-1].data[0][0], mlr_output[-1].data[0][1],
                                      mlr_output[-1].data[0][2], td_error, (exp_x, exp_y))
            teach = to_variable(
                [teach[0], min_max(teach[1], -0.8, 0.8), min_max(teach[2], -0.8, 0.8)])
            mlr.cleargrads()
            loss = F.mean_squared_error(mlr_output[-1], teach)
            loss.backward()
            optimizer.update()
            if state in {'goal', 'out'}:
                if state == 'out':
                    print(f'step(out): {step}')
                    step = para['limit_step'] - 1
                break
        """STEPLOOP_END"""
        """PLOT, SAVE"""
        learning_curve.store(step, offset=0)

        if (epi + 1) % 100 == 0:
            average_step = sum(learning_curve.hanger[0][epi - 99:epi]) / 100
            learning_curve.store(average_step, offset=1)
        print(f'step: {step}, state: {state}\n')
        if (epi % para['plot_cycle'] == 0) or (epi == para['episode'] - 1):
            task.replot2()
            task.replot_goal()
            task.save(f'{database.dirs.data_folders["trajectory"]}ep{epi}')
            reservoir_tau2_output.set_x()
            reservoir_tau10_output.set_x()
            reservoir_tau2_output.plot()
            reservoir_tau10_output.plot()
            reservoir_tau2_output.save(f'{database.dirs.data_folders["ro_tau2"]}{epi}')
            reservoir_tau10_output.save(f'{database.dirs.data_folders["ro_tau10"]}{epi}')
    """EPISODELOOP_END"""
    """PLOT, SAVE"""
    learning_curve.legend = ('step', 'average step')
    learning_curve.set_x(offset=0)
    learning_curve.set_x(range(99, para['episode'], 100), offset=1)
    learning_curve.plot()
    learning_curve.save(f'{database.dirs.data_folders["subroot"]}learning_curve')


def main2():
    database = Database()
    database.dirs.adding_datafolder('trajectory', 'mlr_model', 'ro_tau2', 'ro_tau10')
    para = database.para
    with open(f'{database.dirs.data_folders["subroot"]}mixed_type', 'w') as f:
        f.write(f'')
    with open(f'{database.dirs.data_folders["subroot"]}{para["learning_type"]}', 'w') as f1:
        f1.write(f'')
    np.random.seed(para['seed'])

    task = Field_goal.Field(Num_plot=1, fig_number=0, rew=para['reward'],
                            pena=para['penalty_for_wallcrash'],
                            spena=para['penalty_for_non-switching'], sw_scale=para['switch_flag'])

    reservoir_tau2_tau10 = Past_Reservoir.Reservoir(para['n_neuron']['in'],
                                                    para['n_neuron']['reservoir'],
                                                    para['n_neuron']['fb'],
                                                    v_tau=(10, 2),
                                                    n_tau=(700, 300),
                                                    p_connect=para['connect_rate'],
                                                    v_lambda=para['lambda'],
                                                    in_wscale=para['wscale']['in'],
                                                    fb_wscale=para['wscale']['fb'])

    mlr = Layered_NN.MyChain(para['n_neuron']['reservoir'], para['n_neuron']['in'],
                             para['n_neuron_of_readout']['l1'], para['n_neuron_of_readout']['l2'],
                             para['n_neuron_of_readout']['l3'], para['n_neuron_of_readout']['top'])

    optimizer = optimizers.SGD(lr=para['learning_rate'])
    optimizer.use_cleargrads()
    optimizer.setup(mlr)
    """LOAD MODEL"""
    if para['load_model']:
        serializers.load_npz(para['load_file'], mlr)
        input('model load, ok? Please Enter')
    """GRAPHICAL INSTANCE"""
    learning_curve = Bokeh_line.Line(n_line=2, xlabel='episode', ylabel='step')
    reservoir_tau2_output = Bokeh_line.Line(n_line=10, xlabel='step', ylabel='output(τ=2)')
    reservoir_tau10_output = Bokeh_line.Line(n_line=10, xlabel='step', ylabel='output(τ=10)')

    ma_x, ma_y = 0, 0
    """EPISODELOOP_START"""
    for epi in range(para['episode']):
        print(f'epi: {epi}')
        if epi > 0:
            reservoir_reset(reservoir_tau2_tau10)
        if epi % para['model_savecycle'] == 0 or epi == para['episode'] - 1:
            mlr_save(mlr, f'{database.dirs.data_folders["mlr_model"]}ep{epi}.npz')
        initial_pos_set(task, set_type='random')
        input_signal = task.get_state()
        bypass = to_variable(input_signal)
        input_to_mlr = to_variable(np.hstack(ro_comp(input_signal, reservoir_tau2_tau10)))
        if (epi % para['plot_cycle'] == 0) or (epi == para['episode'] - 1):
            task.ini_add_p_log(posi_x=task.x_agent, posi_y=task.y_agent)
        """STEPLOOP_START"""
        for step in range(para['limit_step']):
            mlr_output = list(mlr.ff_comp(net_in=input_to_mlr, bypass=bypass))
            feedback_to_reservoir_comp(mlr_output[-2].data[0], reservoir_tau2_tau10)
            if para['learning_type'] in ['non-emphasized', 'emphasized']:
                ma_x = moving_average_comp(para['smoothing_factor'], ma_x,
                                           mlr_output[-1].data[0][1])
                ma_y = moving_average_comp(para['smoothing_factor'], ma_y,
                                           mlr_output[-1].data[0][2])
                exp_x = mlr_output[-1].data[0][1] - ma_x
                exp_y = mlr_output[-1].data[0][2] - ma_y
            elif para['learning_type'] == 'conventional':
                exp_x, exp_y = np.random.uniform(-1, 1, (2,))
            else:
                raise Exception(f'Not expected error??????')
            if para['learning_type'] in ['emphasized', 'conventional']:
                task.move_agent(mlr_output[-1].data[0][1] + exp_x,
                                mlr_output[-1].data[0][2] + exp_y)
            elif para['learning_type'] == 'non-emphasized':
                task.move_agent(mlr_output[-1].data[0][1], mlr_output[-1].data[0][2])
            else:
                raise Exception(f'Not expected error!!!!')
            task.agent_crash_wall()
            reward, state = task.state_check()
            """DATASTORE"""
            if (epi % para['plot_cycle'] == 0) or (epi == para['episode'] - 1):
                task.add_p_log2(posi_x=task.x_agent, posi_y=task.y_agent)
                for offset, neuron in enumerate(reservoir_tau2_tau10.ro[0:10]):
                    reservoir_tau2_output.store(neuron, offset=offset)
                for offset, neuron in enumerate(reservoir_tau2_tau10.ro[-10:]):
                    reservoir_tau10_output.store(neuron, offset=offset)
            input_signal = task.get_state()
            bypass = to_variable(input_signal)
            input_to_mlr = to_variable(np.hstack(ro_comp(input_signal, reservoir_tau2_tau10)))
            mlr_pre_output = list(mlr.ff_comp(net_in=input_to_mlr, bypass=bypass))
            if state in {'goal', 'out'}:
                mlr_pre_output[-1] = to_variable([0.0])
            td_error = td_error_comp(para['discount_rate'], reward, mlr_output[-1].data[0][0],
                                     mlr_pre_output[-1].data[0][0])

            teach = teach_signal_comp(mlr_output[-1].data[0][0], mlr_output[-1].data[0][1],
                                      mlr_output[-1].data[0][2], td_error, (exp_x, exp_y))
            teach = to_variable(
                [teach[0], min_max(teach[1], -0.8, 0.8), min_max(teach[2], -0.8, 0.8)])
            mlr.cleargrads()
            loss = F.mean_squared_error(mlr_output[-1], teach)
            loss.backward()
            optimizer.update()
            if state in {'goal', 'out'}:
                if state == 'out':
                    print(f'step(out): {step}')
                    step = para['limit_step'] - 1
                break
        """STEPLOOP_END"""
        """PLOT, SAVE"""
        learning_curve.store(step, offset=0)

        if (epi + 1) % 100 == 0:
            average_step = sum(learning_curve.hanger[0][epi - 99:epi]) / 100
            learning_curve.store(average_step, offset=1)
        print(f'step: {step}, state: {state}\n')
        if (epi % para['plot_cycle'] == 0) or (epi == para['episode'] - 1):
            task.replot2()
            task.replot_goal()
            task.save(f'{database.dirs.data_folders["trajectory"]}ep{epi}')
            reservoir_tau2_output.set_x()
            reservoir_tau10_output.set_x()
            reservoir_tau2_output.plot(store_initialization=True)
            reservoir_tau10_output.plot(store_initialization=True)
            reservoir_tau2_output.save(f'{database.dirs.data_folders["ro_tau2"]}{epi}')
            reservoir_tau10_output.save(f'{database.dirs.data_folders["ro_tau10"]}{epi}')

    """EPISODELOOP_END"""
    """PLOT, SAVE"""
    learning_curve.legend = ('step', 'average step')
    learning_curve.set_x(offset=0)
    learning_curve.set_x(range(99, para['episode'], 100), offset=1)
    learning_curve.plot()
    learning_curve.save(f'{database.dirs.data_folders["subroot"]}learning_curve')


def testing():
    database = Database()
    database.dirs.adding_datafolder('test_trajectory', 'mlr_model', 'ro_tau2', 'ro_tau10')
    para = database.para
    with open(f'{database.dirs.data_folders["subroot"]}testing', 'w') as f:
        f.write(f'')
    with open(f'{database.dirs.data_folders["subroot"]}{para["learning_type"]}', 'w') as f1:
        f1.write(f'')
    np.random.seed(para['seed'])
    field_size = 10
    agent_pos = {'left': (field_size * 1 / 6, field_size / 10),
                 'left_center': (field_size * 2 / 6, field_size / 10),
                 'center': (field_size * 3 / 6, field_size / 10),
                 'right_center': (field_size * 4 / 6, field_size / 10),
                 'right': (field_size * 5 / 6, field_size / 10)}
    goal_pos = (field_size / 2, field_size / 2)
    switch_pos = (field_size / 2, field_size * 9 / 10)

    task = Field_goal.Field(Num_plot=len(agent_pos.keys()), fig_number=0, rew=para['reward'],
                            pena=para['penalty_for_wallcrash'],
                            spena=para['penalty_for_non-switching'], testing=True,
                            sw_scale=para['switch_flag'])

    reservoir_tau2 = Past_Reservoir.Reservoir(para['n_neuron']['in'],
                                              300,
                                              para['n_neuron']['fb'], v_tau=(2,), n_tau=(300,),
                                              p_connect=para['connect_rate'],
                                              v_lambda=para['lambda'],
                                              in_wscale=para['wscale']['in'],
                                              fb_wscale=para['wscale']['fb'])

    reservoir_tau10 = Past_Reservoir.Reservoir(para['n_neuron']['in'],
                                               700,
                                               para['n_neuron']['fb'], v_tau=(10,), n_tau=(700,),
                                               p_connect=para['connect_rate'],
                                               v_lambda=para['lambda'],
                                               in_wscale=para['wscale']['in'],
                                               fb_wscale=para['wscale']['fb'])

    mlr = Layered_NN.MyChain(para['n_neuron']['reservoir'] * 1, para['n_neuron']['in'],
                             para['n_neuron_of_readout']['l1'], para['n_neuron_of_readout']['l2'],
                             para['n_neuron_of_readout']['l3'], para['n_neuron_of_readout']['top'])

    optimizer = optimizers.SGD(lr=para["learning_rate"])
    optimizer.use_cleargrads()
    optimizer.setup(mlr)
    """LOAD MODEL"""
    if para['load_model']:
        serializers.load_npz(para['load_file'], mlr)
        input('model load, ok? Please Enter')

    reservoir_tau2_output = Bokeh_line.Line(n_line=10, xlabel='step', ylabel='output(τ=2)')
    reservoir_tau10_output = Bokeh_line.Line(n_line=10, xlabel='step', ylabel='output(τ=10)')

    ma_x, ma_y = 0, 0
    for pos, epi in enumerate(agent_pos.keys()):
        print(f'ep: {epi}')
        reservoir_reset(reservoir_tau2, reservoir_tau10)
        task.fixation_objects(field_size, field_size, agent_pos[epi], goal_pos, switch_pos)
        task.ini_add_p_log_test(task.x_agent, task.y_agent, var_ep=pos)
        for step in range(para['limit_step']):
            input_signal = task.get_state()
            bypass = to_variable(input_signal)
            input_to_mlr = to_variable(
                np.hstack(ro_comp(input_signal, reservoir_tau2, reservoir_tau10)))
            mlr_output = list(mlr.ff_comp(input_to_mlr, bypass))
            feedback_to_reservoir_comp(mlr_output[-2].data[0], reservoir_tau2, reservoir_tau10)
            if para['learning_type'] in ['non-emphasized', 'emphasized']:
                ma_x = moving_average_comp(para['smoothing_factor'], ma_x,
                                           mlr_output[-1].data[0][1])
                ma_y = moving_average_comp(para['smoothing_factor'], ma_y,
                                           mlr_output[-1].data[0][2])
                exp_x = mlr_output[-1].data[0][1] - ma_x
                exp_y = mlr_output[-1].data[0][2] - ma_y
            elif para['learning_type'] == 'conventional':
                # exp_x, exp_y = np.random.uniform(-1, 1, (2,))
                exp_x, exp_y = 0, 0
            else:
                raise Exception(f'Not expected error??????')
            if para['learning_type'] in ['emphasized', 'conventional']:
                task.move_agent(mlr_output[-1].data[0][1] + exp_x,
                                mlr_output[-1].data[0][2] + exp_y)
            elif para['learning_type'] == 'non-emphasized':
                task.move_agent(mlr_output[-1].data[0][1], mlr_output[-1].data[0][2])
            else:
                raise Exception(f'Not expected error!!!!')
            task.agent_crash_wall()
            reward, state = task.state_check()
            task.add_p_log2(task.x_agent, task.y_agent)
            for offset, neuron in enumerate(reservoir_tau2.ro[0:10]):
                reservoir_tau2_output.store(neuron, offset=offset)
            for offset, neuron in enumerate(reservoir_tau10.ro[0:10]):
                reservoir_tau10_output.store(neuron, offset=offset)
            if state in {'goal', 'out'}:
                break
        """STEPLOOP_END"""
        task.replot_test(var_ep=pos)
        task.replot_goal()
        reservoir_tau2_output.set_x()
        reservoir_tau10_output.set_x()
        reservoir_tau2_output.plot(store_initialization=True)
        reservoir_tau10_output.plot(store_initialization=True)
        reservoir_tau2_output.save(f'{database.dirs.data_folders["ro_tau2"]}{epi}')
        reservoir_tau10_output.save(f'{database.dirs.data_folders["ro_tau10"]}{epi}')
    """EPISODELOOP_END"""
    task.save(f'{database.dirs.data_folders["test_trajectory"]}test')


if __name__ == "__main__":
    main()  # multi-reservoir type
    # main2() # mixed time constant
    # testing()  # multi-reservoir type のテスト
