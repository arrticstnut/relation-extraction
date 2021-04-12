import os
import numpy as np
import random

def read_res(models_results_paths):
    '''    
    #读取目录下各模型的结果
    #model_results_paths: 存放各模型结果的路径字典
    '''
    paths_dict = models_results_paths
    results = {}
    for model_name, filename in paths_dict.items():
            results[model_name] = read_file(filename)
    return results
    

def read_file(filename):
    '''
    #读取文件中的结果
    #filename: 模型结果文件
    '''
    relation_result = {}
    with open (filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.split() # 8065	Entity-Destination(e1,e2)
            #读取结果
            sent_id = line[0] #句子id
            relation_name = line[1] #关系类型名称
            relation_result[sent_id] = relation_name
    return relation_result

def read_labels(relation_file_path):
    '''
    #读取关系标签
    #relation_file_path: 关系标签文件的路径
    '''
    r_labels_name_to_id = {}
    with open (relation_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.split()
            #读取标签
            relation_name = line[1] #关系类型名
            relation_id = line[0] #关系类型id
            r_labels_name_to_id[relation_name] = int(relation_id)
    return r_labels_name_to_id

def get_candidate_relations(results, r_labels_name_to_id):
    '''
    #获得候选关系类型表
    #results：各模型的结果
    #r_labels_name_to_id: 关系标签,{关系名称：关系类型id}
    '''
    #关系类型的总数量
    r_nums = len(r_labels_name_to_id)
    candidate_rels = {}
    for model_name in results:
        model_result = results[model_name]
        for sent_id in model_result:
            if sent_id not in candidate_rels.keys():
                candidate_rels[sent_id] = np.zeros(r_nums)
            #对应坐标的的关系类型数量+1
            relation_id = r_labels_name_to_id[model_result[sent_id]]
            candidate_rels[sent_id][relation_id] += 1
    return candidate_rels

    
def get_prediction_relation(candidate_rels,r_labels_name_to_id):
    '''
    #由于计算score值的输入是考虑了e1和e2的顺序的，但是论文中F1值可以不考虑e1和e2顺序，
    #因此先统计不考虑顺序的结果，在进行预测
    #candidate_rels: 句子的候选关系类型
    #r_labels_name_to_id:关系标签
    '''
    r_nums_bidirec = 19 #考虑方向的关系总类型数量
    r_nums_undirec = 10 #不考虑方法的关系类型总数量
    #统计不考虑顺序的预测关系
    candidate_undirection = {}
    for sent_id, value in candidate_rels.items():
        #获取value中奇数位置的值
        #candidate_undirection[sent_id] = value[0::2]
        candidate_undirection[sent_id] = [v for v in value][0::2]
        #将value中偶数位置的值加入
        for i in range(1,len(value),2):
            candidate_undirection[sent_id][i // 2] += value[i]

    #从不考虑顺序的预测关系中选出最大值
    rels_undirec = {}
    for sent_id, value in candidate_undirection.items():
        #value中的最大值
        max_v = max(value)
        #获得最大值所在的下标
        r_indexs = [i for i,v in enumerate(value) if v == max_v]
        #从多个最大值中随机选择一个
        rels_undirec[sent_id] = r_indexs[random.randint(0,len(r_indexs)-1)]
    
    #还原到考虑顺序的预测类型表中
    rels = {}
    for sent_id, r in rels_undirec.items():
        #得到无顺序到有顺序的关系的对应id
        r1,r2 = r*2, r*2+1 
        #选取值较大的关系id
        rels[sent_id] = r2 if (r2 < r_nums_bidirec and candidate_rels[sent_id][r2] > candidate_rels[sent_id][r1]) else r1
    return rels

def write_res(rels, r_labels, results_path):
    with open (results_path, 'w', encoding='utf-8') as f:
        for sent_id, r in rels.items():
            line = str(sent_id) + '\t' + str(r_labels[r])
            f.write(line + '\n')
    return True

def vote(models_results_paths, relation_file_path, out_path):
    '''
    #input_dir：存放各模型的结果的目录
    #relation_file_path: 存放关系标签的路径
    #out_path: 输入文件的路径
    '''
    #读取各模型的结果
    results = read_res(models_results_paths)
    #读取关系标签
    r_labels_name_to_id = read_labels(relation_file_path)
    #获得候选关系
    candidate_rels = get_candidate_relations(results,r_labels_name_to_id)
    #打分确定最终关系
    rels = get_prediction_relation(candidate_rels,r_labels_name_to_id)
    #写入文件
    r_labels_id_to_name = {r_id:r_name for r_name, r_id in r_labels_name_to_id.items()}
    write_res(rels, r_labels_id_to_name, out_path)

if __name__ == '__main__':
    paths_dict = {
        'cnn': '../cnn+pos+win345/out/results.txt',
        'blstm': '../blstm+pos+attention/out/results.txt',
        'series-cnn-blstm': '../series-cnn-blstm/out/results.txt',
        'series-blstm-cnn': '../series-blstm-cnn/out/results.txt',
        'parallel-cnn-blstm': '../parallel-cnn-blstm/out/results.txt',
    }
    relation_file_path = '../data/SemEval_2010_task_8/relations.txt'
    out_path = './out/results.txt'
    vote(paths_dict, relation_file_path, out_path)