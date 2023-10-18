import json
import pandas as pd
from conclude_detector import init_conclude_model, conclude_detector

# set paths
pattern_path = 'data/conclude_pattern_121.txt'
model_path = 'data/gbdt_conclude_detector.plk'
patterns = open(pattern_path, 'r').read().strip().split('\n')
embedding_path = 'data/qq_w2v.pickle'
pattern2index_path = 'data/conclude_pattern2index.plk'
# init conclude_model
conclude_model = init_conclude_model(
    model_path, patterns, pattern2index_path, embedding_path)
# load text_list
df_course = pd.read_excel('test_data/demo.xlsx')
text_list = json.loads(
    df_course[['begin_time', 'end_time', 'text']].to_json(orient='records'))
# 调用检测
def process_text(text_list, conclude_model):
    # 使用模型对文本列表进行处理
    result = conclude_detector(text_list,conclude_model)
    
    # 打印结果
    print('result:', result)
    
    # 计算并打印结果的长度
    len_result = len(result)
    print('Length of result:', len_result)

    # 计算并打印原始文本列表的长度
    len_text_list = len(text_list)
    print('Length of text list:', len_text_list)

     # 计算两者的比例 
    ratio = 0 if len_text_list == 0 else float(len_result) / float(len_text_list)

    return ratio

# 调用函数示例：
ratio = process_text(text_list, conclude_model)
print(ratio)