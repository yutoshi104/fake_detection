import os
import numpy as np
from pprint import pprint


# 繰り返し実行させたいファイル
file_list = ["retrain.sh","retrain_rnn.sh"]

# qstat情報取得
qstat = os.popen('qstat').read().split('\n')[2:-1]
for i in range(len(qstat)):
    qstat[i] = qstat[i].split()
qstat = np.array(qstat)
pprint(qstat)
print(qstat[:, 2])










exit()
os.system('qsub train.sh')

