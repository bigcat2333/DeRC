import os

file_type = 'qqp'
log_path = "/path/to/save/checkpoints/DeRC_layer3_QQP/"

config = {'hans':{'dev':'eval results on mnli_dev_m:', 'challenge':'eval results on hans:'}, 'qqp':{'dev':'eval results on qqp_dev:', 'challenge':'eval results on paws:'}}

file_list = os.listdir(log_path)
index_list = [int(x.replace('model_','')) for x in file_list]
index_list.sort()
for index in index_list:
    # print(index)
    file_name = 'model_' + str(index)
    file_path = os.path.join(log_path, file_name, 'eval_all_results.txt')
    with open(file_path, encoding='utf-8') as f:
        lines = [line.replace('\n','') for line in f.readlines()]
        # dev_index = lines.index(config[file_type]['dev']) + 2
        challenge_index = lines.index(config[file_type]['challenge']) + 2
        # dev_line = lines[dev_index].replace('acc =', '').replace('\n', '')
        challenge_line = lines[challenge_index].replace('acc =', '').replace('\n', '')
        print(file_name)
        print(challenge_line)