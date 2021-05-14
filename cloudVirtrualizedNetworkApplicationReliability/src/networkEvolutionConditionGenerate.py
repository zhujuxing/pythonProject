"""

"""
import math
import random
import re
import time

import networkx as nx
import numpy as np
import pandas as pd

import networkEvolutionObjectModel
from networkEvolutionObjectModel import CloudVirtualizedNetwork


def generate_network_evolution_condition(gpath, T):
    T, cdf = read_data(gpath, T)
    cdf1 = generate_failure_state_multi_component_single_failuremode(cdf, T)
    cdf1 = generate_failure_state_multi_component_multi_failuremode(cdf1)
    time_set = generate_time_point_set_of_evolution_condition(cdf1)
    evol = generate_evolution_condition_raw(cdf1, T, time_set)
    evol = format_evol(evol)
    return evol


def time_format(x: str) -> float:
    """
    转换时间单位的函数，将所有时间单位统一为小时
    """
    result = 0.0
    if isinstance(x, str):
        s = re.findall("\d+", x)[0]
        time = float(s)
        if ('年' in x) or ('y' in x):
            result = 365 * 24 * time
        elif 'min' in x:
            result = time / 60
        elif 's' in x:
            result = time / 3600
        elif 'h' in x:
            result = time
    return result


def is_hardware(x: str) -> bool:
    if x in ('DCGW', 'Server', 'EOR', 'TOR', 'Edge', 'PL'):
        return True
    else:
        return False


def sample_time_form_distribution(mean_val: float, distribution: str, seed=None) -> float:
    if not seed:
        delta = random.random()
    elif type(seed) == float:
        if (seed < 1) and (seed > 0):
            delta = seed
        else:
            raise Exception("给定种子应在[0,1]之间")
    else:
        raise Exception("给定种子格式不正确")

    sample_time = 0.0
    if distribution == "常数":
        sample_time = mean_val
    elif distribution == "均匀型":
        pass
    elif distribution == "正态型":
        pass
    elif distribution == "指数型":
        sample_time = - math.log(delta) * mean_val
    elif distribution == "指数故障间隔时间":
        pass
    else:
        raise Exception("未定义此类分布。")
    return sample_time


def read_data(data: str or CloudVirtualizedNetwork, T: int) -> (int, pd.DataFrame):
    time2hour_T = T * 365 * 24
    if type(data) == str:
        if data.endswith(".gpickle"):
            g = nx.read_gpickle()
        else:
            raise Exception("在进行演化条件生成时，读取的外部文件格式错误。")
    elif type(data) == CloudVirtualizedNetwork:
        g = data
    else:
        raise Exception("在进行演化条件生成时，读取的外部文件格式错误。")

    ndf = g.graph['node_info']
    edf = g.graph['edge_info']
    fdf = g.graph['fail_info']

    ndf = ndf.merge(fdf, on=['Type'])
    edf = edf.merge(fdf, on=['Type'])
    ndf["FailureTime"] = 0
    ndf["RepairTime"] = 0
    edf["FailureTime"] = 0
    edf["RepairTime"] = 0
    ndf = ndf.rename(columns={"NodeID": "ID"})
    edf = edf.rename(columns={"EdgeID": "ID"})
    ndf = ndf[
        ['ID', 'Type', 'FailType', 'MTBF', 'FailDistri', 'FDR', 'FDT', 'FDPT', 'FARR', 'FART', 'MTTR', 'RecoDistri',
         'RecoStra']].copy(deep=True)
    edf = edf[
        ['ID', 'Type', 'FailType', 'MTBF', 'FailDistri', 'FDR', 'FDT', 'FDPT', 'FARR', 'FART', 'MTTR', 'RecoDistri',
         'RecoStra']].copy(deep=True)
    cdf = ndf.append(edf)
    return time2hour_T, cdf


def generate_failure_state_single_component_single_failuremode(x, T, cdf):
    '''
    单个构件、单个模式生成状态
    Parameters
    ----------
    x : df行值

    Returns
    -------
    fail_time : 单个构件单个模式失效列表
    reco_time : 单个构件单个模式恢复列表

    '''
    fail_dt = cdf[cdf['ID'] == x['ID']]
    fail_time = []
    reco_time = []
    for row in fail_dt.iterrows():
        t = 0
        MTBF = time_format(row[1]['MTBF'])
        FDR = row[1]['FDR']
        FDT = time_format(row[1]['FDT'])
        FARR = row[1]['FARR']
        FART = time_format(row[1]['FART'])
        MTTR = time_format(row[1]['MTTR'])
        fail_distribution = row[1]['FailDistri']
        recover_distribution = row[1]['RecoDistri']
        fdpt = time_format(row[1]['FDPT'])

        fail_time_list = []
        reco_time_list = []
        while True:
            if t > T:
                break
            else:
                t_f = sample_time_form_distribution(MTBF, fail_distribution)
                t_hr = sample_time_form_distribution(MTTR, recover_distribution)
                t_r = cal_repair_time(FARR, FART, FDR, FDT, fdpt, row, t_hr, period_check=True)

                if (t + t_f) <= T:
                    fail_time_list.append(t + t_f)
                else:
                    break
                if (t + t_f + t_r) <= T:
                    reco_time_list.append(t + t_f + t_r)
                else:
                    break
                t = t + t_f + t_r
        fail_time.append(fail_time_list.copy())
        reco_time.append(reco_time_list.copy())
    return fail_time, reco_time


def cal_repair_time(FARR, FART, FDR, FDT, fdpt, row, t_hr, period_check=False):
    if period_check:
        t_r = 0
        flag_failure_detect = False
        while not flag_failure_detect:
            if random.random() < FDR:
                flag_failure_detect = True
                t_r += FDT
                if is_hardware(row[1]['Type']):
                    t_r += t_hr
                else:
                    if random.random() < FARR:
                        t_r += FART
                    else:
                        t_r += t_hr
            else:
                t_r += fdpt
    else:
        # 首先看是否检测到故障
        if random.random() < FDR:
            t_r = FDT
            # 对于硬件节点，维修时间即为叫人维修的时间
            if is_hardware(row[1]['Type']):
                t_r += t_hr
            # 对于软件节点，维修时间需要根据是否自动维修确定
            else:
                # 自动维修
                if random.random() < FARR:
                    t_r += FART
                # 人工维修
                else:
                    t_r += t_hr
        else:
            if is_hardware(row[1]['Type']):
                t_r = fdpt
            else:
                t_r = fdpt
    return t_r


def generate_failure_state_multi_component_single_failuremode(cdf: pd.DataFrame,
                                                              T: float) -> pd.DataFrame:
    cdf_temp = cdf.apply(lambda x: generate_failure_state_single_component_single_failuremode(x, T, cdf), axis=1)
    cdf['FailureTime'] = cdf_temp.apply(lambda x: x[0])
    cdf['RepairTime'] = cdf_temp.apply(lambda x: x[1])
    cdf1 = cdf[['ID', 'FailureTime', 'RepairTime']].copy(deep=True)
    cdf1.drop_duplicates(['ID'], inplace=True, ignore_index=True)
    return cdf1


def generate_failure_state_single_component_multi_failuremode(x):
    fail_time = []
    reco_time = []
    for i in x['FailureTime']:
        fail_time.extend(i)
    fail_time.sort()
    for i in x['RepairTime']:
        reco_time.extend(i)
    reco_time.sort()
    return fail_time, reco_time


def generate_failure_state_multi_component_multi_failuremode(cdf1):
    temp = cdf1.apply(generate_failure_state_single_component_multi_failuremode, axis=1)
    cdf1['FailureTime'] = temp.apply(lambda x: x[0])
    cdf1['RepairTime'] = temp.apply(lambda x: x[1])
    return cdf1


def generate_time_point_set_of_evolution_condition(cdf1):
    time_set = set([])
    for i in cdf1['FailureTime']:
        time_set = time_set | set(i)
    for i in cdf1['RepairTime']:
        time_set = time_set | set(i)
    time_set = [i for i in time_set]
    time_set.sort()
    print('演化态共有%d个' % len(time_set))
    return time_set


def generate_evolution_condition_raw(cdf1, T, time_set):
    t = 0
    ftl = cdf1['FailureTime'].to_list()
    rtl = cdf1['RepairTime'].to_list()
    all_fail_component_set = []
    all_recovcer_component_set = []
    while t <= T:
        fail_time_list = [i[0] if i != [] else T for i in ftl]
        recover_time_list = [i[0] if i != [] else T for i in rtl]
        fail_time = min(fail_time_list)
        recover_time = min(recover_time_list)
        fail_component_index = fail_time_list.index(fail_time)
        recover_component_index = recover_time_list.index(recover_time)
        fail_set = []
        recover_set = []
        if fail_time < recover_time:
            fail_set.append(cdf1.loc[fail_component_index, 'ID'])
            t = fail_time
            ftl[fail_component_index] = ftl[fail_component_index][1:]
        elif fail_time > recover_time:
            recover_set.append(cdf1.loc[recover_component_index, "ID"])
            t = recover_time
            rtl[recover_component_index] = rtl[recover_component_index][1:]
        else:
            break
        all_fail_component_set.append(fail_set)
        all_recovcer_component_set.append(recover_set)
    evol = pd.DataFrame([all_fail_component_set, all_recovcer_component_set])
    evol = evol.T
    evol_time = [time_set[i + 1] - time_set[i] for i in range(len(time_set) - 1)]
    evol_time.append(T - time_set[-1])
    evol.columns = ['EvolFailComponentsSet', 'EvolRecoComponentsSet']
    try:
        evol.insert(0, 'EvolTime', evol_time)
    except:
        evol.insert(0, 'EvolTime', evol_time[:len(evol)])
    return evol


def format_evol(evol):
    t = 0

    def time_add(x):
        nonlocal t
        result = [t, t + x]
        t += x
        return result

    evol['EvolTime'] = evol['EvolTime'].apply(time_add)

    fail_components_set = []

    def to_fail_components_set(x):
        nonlocal fail_components_set
        if len(x['EvolRecoComponentsSet']) == 0:
            fail_components_set.append(x['EvolFailComponentsSet'][0])
        else:
            fail_components_set.remove(x['EvolRecoComponentsSet'][0])
        return fail_components_set.copy()

    evol['EvolFailComponentsSet'] = evol.apply(to_fail_components_set, axis=1)
    return evol


# --------------------------测试函数区-----------------------------------
def test_time_format():
    x = ['1y', '1年', '1min', '1s', '1h']
    for i in x:
        print(time_format(i))


def test_read_data():
    g = networkEvolutionObjectModel.test()
    t, cdf = read_data(g, 10)
    print(t)
    print(cdf)


def test_sample_time_form_distribution():
    mean_val = 50
    distribution = "指数型"
    print(sample_time_form_distribution(mean_val, distribution))


def test_generate_failure_state_single_component_single_failuremode():
    T = 100
    g = networkEvolutionObjectModel.test()
    ndf = g.graph["node_info"]
    fdf = g.graph["fail_info"]
    ndf = pd.merge(ndf, fdf, on="Type")
    ndf = ndf.rename(columns={'NodeID': "ID"})
    T, cdf = read_data(g, T)
    x = cdf.iloc[0]
    return generate_failure_state_single_component_single_failuremode(x, T, ndf)


def test_generate_network_evolution_condition():
    t_start = time.time()
    T = 100
    # gpath = os.path.abspath(os.path.dirname(os.getcwd()) + os.path.sep + '.') + os.sep + ''
    gpath = networkEvolutionObjectModel.test()
    evol = generate_network_evolution_condition(gpath, T)
    t_end = time.time()
    print(t_end - t_start)
    evol.to_excel('e.xlsx')
    return evol


def test_single_componet_multi_failmode_multitimes():
    T = 10000
    N = 10
    g = networkEvolutionObjectModel.test()
    T, cdf = read_data(g, T)
    mtbf_df = pd.DataFrame(columns=list(range(N)))
    component_df = cdf[['ID', 'Type', 'MTBF', 'MTTR']]
    for i in range(N):
        test_component_info = generate_failure_state_multi_component_single_failuremode(cdf, T)
        temp = test_component_info.apply(lambda x: test_MTBF(x, T), axis=1)
        mtbf_df[i] = temp
    component_df['MTBF_CAL'] = mtbf_df.apply(np.mean, axis=1)
    print(component_df)


def test_MTBF(x, T):
    try:
        timelist = x['FailureTime'][0]
    except:
        timelist = x['FailureTime']
    timelist.append(T)
    time_end = np.array(timelist)
    time_start = timelist[:-1]
    time_start.insert(0, 0)
    time_start = np.array(time_start)
    test_MTBF_val = np.mean(time_end - time_start)
    return test_MTBF_val / 24 / 365


if __name__ == '__main__':
    # test_time_format()
    # test_read_data()
    # test_sample_time_form_distribution()
    # evol = test_generate_failure_state_single_component_single_failuremode()
    evol = test_generate_network_evolution_condition()
    # test_single_componet_multi_failmode_multitimes()
