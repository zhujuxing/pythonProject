import copy

import pandas as pd

import networkEvolutionObjectModel
from networkEvolutionObjectModel import CloudVirtualizedNetwork


def network_evolutin_rules_analysis_for_evolution_condition(g: CloudVirtualizedNetwork,
                                                            evol_input: pd.DataFrame or str) -> CloudVirtualizedNetwork:
    g_t = copy.deepcopy(g)
    up_time_dict = {}
    down_time_dict = {}
    up_list = []
    down_list = []
    evol = pd.DataFrame()
    if type(evol_input) == str:
        try:
            evol = pd.read_excel(evol_input)
            evol['EvolTime'] = evol['EvolTime'].apply(eval)
            evol['EvolFailComponetsSet'] = evol['EvolFailComponetsSet'].apply(eval)
            evol['EvolRecoComponetsSet'] = evol['EvolRecoComponetsSet'].apply(eval)
        except:
            raise Exception("输入文件格式错误")
    elif type(evol_input) == pd.DataFrame:
        evol = evol_input
    else:
        raise Exception("输入的演化条件应为Dataframe类型或者响应的.xlsx文件")
    for evol_each_time in evol.iterrows():
        x = evol_each_time[1]
        g_t = network_evolution_rules_anlysis_for_single_evolution_state(g_t, x)
    return g_t


def network_evolution_rules_anlysis_for_single_evolution_state(g: CloudVirtualizedNetwork,
                                                               x) -> CloudVirtualizedNetwork:
    return g


def update_application_to_components(g: CloudVirtualizedNetwork):
    ndf = g.graph['node_info']
    edf = g.graph['edge_info']
    ndf = ndf.rename(columns={"NodeID": "ID"})
    edf = edf.rename(columns={"EdgeID": "ID"})
    # ndf = ndf[["ID", "Type"]]
    # edf = edf[["ID", "Type"]]
    ndf = ndf[['Type']]
    edf = edf[['Type']]
    # cdf = pd.merge(ndf, edf, how='outer')
    cdf = ndf.append(edf)
    vdf = g.graph['vnf_info']

    # cdf['VNF'] = []
    # cdf['Application'] = []
    return cdf


# --------------------------测试函数区-----------------------------------
def test_network_evolutin_rules_analysis_for_evolution_condition():
    g = networkEvolutionObjectModel.test()
    evol = pd.DataFrame()
    network_evolutin_rules_analysis_for_evolution_condition(g, evol_input=evol)


def test_update_application_to_components():
    g = networkEvolutionObjectModel.test()
    cdf = update_application_to_components(g)
    return cdf


if __name__ == '__main__':
    test_network_evolutin_rules_analysis_for_evolution_condition()
    # cdf = test_update_application_to_components()