# -*- coding:utf-8 -*- 

import itertools as it
from collections import namedtuple, defaultdict

from pylab import *

import Form as form


def append_data(datalist: tuple, apd: tuple):
    if len(datalist) != len(apd):
        raise AttributeError("len(datalist):{} and len(apd):{} must make equal".format(len(datalist), len(apd)))
    for len_data in range(len(datalist)):
        datalist[len_data].append(apd[len_data])


def input_word(crash_word: str, print_word: str = "crash_word, random or ('x', 'y')"):
    word = input(print_word)
    if word == crash_word:
        sys.exit(1)
    else:
        return word
    

def p_judge(stkp: tuple, stkr: tuple, jdgp: tuple, jdgr: float) -> str:  # stk : stock
    if len(stkp) != len(stkr):
        raise Exception("len arg(obp):{} and arg(obr):{} must be equal".format(len(stkp), len(stkr)))
    dist, sum_r = [], []
    for jdg_p, stk_p in it.product((jdgp,), stkp):
        print(jdg_p, stk_p)
        sq_dx, sq_dy = jdg_p[0] - stk_p[0], jdg_p[1] - stk_p[1]  # sq : squaring
        sq_dx **= 2
        sq_dy **= 2
        dist.append(math.sqrt(sq_dx + sq_dy))
    for jdg_r, stk_r in it.product((jdgr,), stkr):
        print(jdg_r, stk_r)
        sum_r.append(jdg_r + stk_r)
    for len_dist in range(len(dist)):
        print(dist)
        print(sum_r)
        if dist[len_dist] <= sum_r[len_dist]:
            # print("error position between objects, offset")
            return "error init position"
        else:
            if len_dist == len(dist) - 1:
                return "correct init position"
            else:
                continue


class Field:
    def __init__(self, fdsize=(10, 10), n_fig=0, figsize=(6, 6), n_fm=111):
        self.fdsize = fdsize  # fd:field
        self.fig = plt.figure(num=n_fig, figsize=figsize)
        self.ax = self.fig.add_subplot(n_fm)
        axis([0, fdsize[0], 0, fdsize[1]])
        xlabel("x_axis")
        ylabel("y_axis")
        title("Simulation")
        legend()
        grid(True)

    def __call__(self, radi):
        """
        objectsが配置可能なフィールドの範囲を実装する関数
        :param radi:
        :return:
        """
        lwx, upx = radi, self.fdsize[0] - radi  # upx:upper_x, lwx:lower_x
        lwy, upy = radi, self.fdsize[1] - radi
        print((lwx, upx), (lwy, upy))
        return (lwx, upx), (lwy, upy)


class Objects:
    def __init__(self, nm_ob: tuple, obp: tuple, obr: tuple, obc: tuple, fd=Field(), frm=form.circle):
        self.fd = fd
        self.frm = frm
        self.nm_ob = nm_ob
        self.p = obp
        self.r = obr
        self.list_ob = defaultdict(lambda: "none(nm_ob) in class : {}".format(self.__class__.__name__))
        for len_nm_ob in range(len(nm_ob)):
            if obp[len_nm_ob] == "random":
                xscale, yscale = self.fd(obr[len_nm_ob])
                p = self.set_posi(xscale, yscale, random=True)
            else:
                p = obp[len_nm_ob]
            self.list_ob[nm_ob[len_nm_ob]] = (p, obr[len_nm_ob], obc[len_nm_ob])

    def get_list_ob(self, len_nm_ob):
        return self.list_ob[self.nm_ob[len_nm_ob]][0], self.list_ob[self.nm_ob[len_nm_ob]][1], \
               self.list_ob[self.nm_ob[len_nm_ob]][2]

    def set_posi(self, xscale: tuple, yscale: tuple, random: bool = False, p: float = ("x", "y")):
        if random:
            x = np.random.uniform(xscale[0], xscale[1])
            y = np.random.uniform(yscale[0], yscale[1])
        else:
            x, y = p
        return x, y


class Agent(Objects):
    def __init__(self, nm_a: tuple = ("ag1",), ap: tuple = ("random",), ar: tuple = (0.5,), ac: tuple = ("red",),
                 fd=Field()):
        # super().__init__(n_ob=n_a, nm_ob=nm_a, obp=ap, obr=ar, obc=ac)
        super().__init__(nm_ob=nm_a, obp=ap, obr=ar, obc=ac, fd=fd)
        self.xscale = None

    def move(self, dx, dy):
        self.lx, self.ly = (self.x, self.y)  # a_lx:agent_last_x
        self.x += dx
        self.y += dy

    def wall_judge(self):
        wx, wy = "field", "field"
        if not self.xscale[0] < self.x < self.xscale[1]:
            self.x = self.lx
            wx = "wall_x"
        if not self.yscale[0] < self.y < self.yscale[1]:
            self.y = self.ly
            wy = "wall_y"
        return wx, wy


class Goal(Objects):
    def __init__(self, nm_g: tuple = ("g1",), gp: tuple = ("random",), gr: tuple = (1,), gc: tuple = ("blue",),
                 fd=Field()):
        # super().__init__(n_ob=n_g, nm_ob=nm_g, obp=gp, obr=gr, obc=gc)
        super().__init__(nm_ob=nm_g, obp=gp, obr=gr, obc=gc, fd=fd)


class Switch(Objects):
    def __init__(self, nm_s: tuple = ("g1",), sp: tuple = ("random",), sr: tuple = (1,), sc: tuple = ("blue",),
                 fd=Field()):
        """
        Switchの機能をするクラス。
        :param sp: position of switch "random" の場合,初期位置をランダムに決定してくれる
        :param sr: radius of switch
        :param sc: color of switch
        :param fd: Field()
        """
        # super().__init__(n_ob=n_s, nm_ob=nm_s, obp=sp, obr=sr, obc=sc)
        super().__init__(nm_ob=nm_s, obp=sp, obr=sr, obc=sc, fd=fd)


class Trial:
    def __init__(self, names: tuple, *objs: object, field=Field()):
        """
        タスク関係の試行全般のクラス。
        :param names: name of objects
        :param objs: objects(agent, goal and so on)
        :param field: Field() class is absolutely required
        """
        if len(names) != len(objs) + 1:
            raise SyntaxError('len(names): {} != len(layers): {}'.format(len(names), len(objs) + 1))
        self.fd = field
        self.data = defaultdict(lambda: "none(object) in class {}".format(self.__class__.__name__))
        stock_p, stock_r = [], []
        for len_obj in range(len(objs)):
            self.data[names[len_obj]] = objs[len_obj]  # data : dictionary for objects of Agent(), Goal() and so on
            # print(self.data)
            nm_ob = self.data[names[len_obj]].nm_ob
            # print(nm_ob)
            for len_nm_ob in range(len(nm_ob)):
                # print(self.data[names[len_obj]].__class__)
                # print(self.data[names[len_obj]].get_list_ob(len_nm_ob))
                p, r, c = self.data[names[len_obj]].get_list_ob(len_nm_ob)
                # print("p : {}".format(p))
                # print("r : {}".format(r))
                # print("c : {}".format(c))
                # print("len_obj : {}, len_nm_ob : {}".format(len_obj, len_nm_ob))
                while True:
                    if [len_obj, len_nm_ob] == [0, 0]:
                        form = self.data[names[len_obj]].frm(center_point=p, radius=r, facecolor=c, edgecolor="black")
                        self.fd.ax.add_patch(form)
                        break
                    else:
                        # print("p : {}".format(p))
                        # print("r : {}".format(r))
                        # print("c : {}".format(c))
                        # print("stock_p : {}".format(stock_p))
                        # print("stock_r : {}\n".format(stock_r))
                        judge = p_judge(tuple(stock_p), tuple(stock_r), p, r)
                        # print(judge)
                        if judge == "error init position":
                            word = "random"
                            # word = input_word(crash_word="crash",  print_word="\nPlease enter crash_word, random or posi : ")
                            xscale, yscale = self.fd(r)
                            if word == "random":
                                p = self.data[names[len_obj]].set_posi(xscale, yscale, random=True)
                            elif word == "posi":
                                p = float(input("x : ")), float(input("y :"))
                                p = self.data[names[len_obj]].set_posi(xscale, yscale, p=p)
                                # print(p)
                            else:
                                pass
                            continue
                        else:
                            form = self.data[names[len_obj]].frm(center_point=p, radius=r, facecolor=c,
                                                                 edgecolor="black")
                            self.fd.ax.add_patch(form)
                            break
                append_data((stock_p, stock_r), (p, r))
                # print("stock_p : {}".format(stock_p))
                # print("stock_r : {}\n\n".format(stock_r))


if __name__ == "__main__":
    for loop in range(100):
        np.random.seed(loop + 800)
        name = ("ag", "gl", "sw", "fd")
        Data = namedtuple("Obj", name)
        field = Field(fdsize=(3, 3))
        data = Data(ag=Agent(("a1", "a2"), ("random", "random"), (0.1, 0.5), ("red", "magenta"), fd=field),
                    gl=Goal(("g",), ("random",), (0.1,), ("blue",), fd=field),
                    sw=Switch(("s",), ("random",), (0.5,), ("yellow",), fd=field),
                    fd=field)
        trl = Trial(name, data.ag, data.gl, data.sw, field=data.fd)  # ここでフィールド完成
        # plt.show()
        plt.pause(3)
        plt.close()
