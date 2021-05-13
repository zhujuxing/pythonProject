"""

"""

import os

import networkx as nx
import pandas as pd


class CloudVirtualizedNetwork(nx.Graph):

    def __init__(self, file: str):
        """
        该函数为演化对象生成函数。

        Parameters
        ----------
        file : str
            网络信息.xlsx文件的路径。

        Returns
        -------
        G : TYPE
            G包含节点信息:G.nodes.data(),链路信息G.edges.data(),
            业务信息在G.graph里设置，分别是
            G.graph['VNF_info'] : dataframe对象
            G.graph['Service_info']
            G.graph['Application_info']

        """
        # super().__init__(**attr)
        nx.Graph.__init__(self)
        node_info = pd.read_excel(file, sheet_name='node_info')
        edge_info = pd.read_excel(file, sheet_name='edge_info')
        fail_info = pd.read_excel(file, sheet_name='fail_info')
        vnf_info = pd.read_excel(file, sheet_name='vnf_info')
        application_info = pd.read_excel(file, sheet_name=' application_info')

        self.add_nodes_from(node_info['节点ID'])
        egs = edge_info[['源节点ID', '目的节点ID']].to_numpy().tolist()
        self.add_edges_from(egs)

        node_info = node_info.rename(columns={"节点ID": 'NodeID',
                                              "节点类型": "Type",
                                              "节点功能": "NodeFunction"
                                              })
        edge_info = edge_info.rename(columns={"链路ID": 'EdgeID',
                                              "源节点ID": 'EdgeSourceNode',
                                              "目的节点ID": 'EdgeDestinationNode',
                                              "链路类型": 'Type'
                                              })
        fail_info = fail_info.rename(columns={'类型': 'Type',
                                              '故障模式': 'FailType',
                                              '平均故障间隔时间': 'MTBF',
                                              '故障时间分布': 'FailDistri',
                                              '故障检测率': 'FDR',
                                              '故障检测时间': 'FDT',
                                              '故障重新检测时间': 'FDPT',
                                              '自动维修概率': 'FARR',
                                              '自动维修时间': 'FART',
                                              '平均人工维修时间': 'MTTR',
                                              '维修时间分布': 'RecoDistri',
                                              '备份策略': 'RecoStra'
                                              })
        vnf_info = vnf_info.rename(columns={'VNF名称': 'VNFID',
                                            '数据类型': 'VNFDataType',
                                            '备份类型': 'VNFBackupType',
                                            '工作节点': 'VNFDeployNode',
                                            '备用节点': 'VNFBackupNode',
                                            '倒换概率': 'VNFFailSR',
                                            '倒换时间': 'VNFFailST',
                                            '倒换控制链路': 'VNFSwitchPath'
                                            })
        application_info = application_info.rename(columns={'业务名称': 'ApplicationID',
                                                            '业务逻辑路径': 'ApplicationVNFs',
                                                            '业务物理路径': 'ApplicationWorkPath',
                                                            '业务中断时间': 'ApplicationUnavailTime'
                                                            })

        node_info['State'] = 1
        node_info['Idle'] = [1 if i == "空闲" else 0 for i in node_info['NodeFunction']]

        self.graph['node_info'] = node_info
        self.graph['edge_info'] = edge_info
        self.graph['vnf_info'] = vnf_info
        self.graph['application_info'] = application_info
        self.graph['fail_info'] = fail_info

        print("网络对象已生成")

    def __str__(self):
        string = "----------------节点信息---------------\n" \
                 + str(self.graph['node_info']) + "\n"\
                 + "----------------链路信息---------------\n" \
                 + str(self.graph['edge_info']) + "\n"\
                 + "----------------故障信息---------------\n" \
                 + str(self.graph['fail_info']) + "\n"\
                 + "----------------vnf信息---------------\n" \
                 + str(self.graph['vnf_info']) + "\n"\
                 + "----------------业务信息---------------\n" \
                 + str(self.graph['application_info']) + "\n"
        return string

    def __del__(self):
        print("网络对象已销毁")


def test():
    # file =
    file = os.path.abspath(os.path.dirname(os.getcwd()) + os.path.sep + ".") + os.sep + 'inputFiles' + os.sep + \
           'case_8server_2link_backup.xlsx'
    return CloudVirtualizedNetwork(file)


if __name__ == '__main__':
    g = test()
    print(g)
