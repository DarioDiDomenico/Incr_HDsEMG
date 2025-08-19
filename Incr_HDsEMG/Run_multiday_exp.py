import yaml
import Multiday_batch, Multiday_incr, Multiday_LDA, Multiday_kNN
import numpy as np
import time
import os
from datetime import datetime
from Data_Handler import read_params_file, batch_comb
from itertools import permutations
start_time = time.time()

def write_log(file_name, exp_time, parameters):
    results_path = os.path.join(parameters['save_folder'],file_name.split(os.sep)[1],'multiday',parameters['ML_model'])
    if not os.path.exists(results_path):
        os.makedirs(results_path)
        # Program to append to text file using write() function
    with  open(os.path.join(results_path,'logs_mutliday_exp_'+parameters['ML_script'].split('_')[1]+'.txt'), "a") as file:
        file.write('date: '+ datetime.today().strftime('%Y-%m-%d_%H:%M:%S"'))
        file.write('\t')
        file.write('exp_time: '+ str(np.around(exp_time,2)))
        file.write('\t')
        file.write('exp: '+ '-'.join(file_name.split(os.sep)[1:-1]))
        file.write('\t')
        file.write('ML_model: '+ parameters['ML_model'])
        file.write('\t')
        file.write('\n')
        file.close()

def main(parameters):
    if parameters['ML_script'] == 'Multiday_incr':
        perm_list = list(permutations(parameters['file_name']))
    else:
        perm_list = batch_comb(parameters['file_name'])
        
    for perm_iter in range(len(perm_list)):
        single_exp_time = time.time()
        perm = perm_list[perm_iter]
        print('############STARTING############')
        perm_str = []
        [perm_str.append(x.split(os.sep)[2]) for x in perm]
        print('Permutation: ', perm_str)
        eval(parameters['ML_script']+'.main(parameters, perm)')
        
        print("The permutation exp on", perm_str,' took ', "{0:.2f}".format(time.time()-single_exp_time), "s")
        if parameters['save_results']:
            write_log(parameters['file_name'][0], time.time()-single_exp_time, parameters)        
    
if __name__ == '__main__':
    opt = read_params_file()
    f = open(opt.params_file,'rb')
    parameters = yaml.load(f, Loader=yaml.FullLoader)
    main(parameters)
    print("The program took", "{0:.2f}".format(time.time()-start_time), "seconds to run")


