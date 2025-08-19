import Run_single_exp
import Run_multiday_exp
import os
import subprocess
import time


def main():
    run_single_exp = True
    run_multiday_exp = True
    params_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'parameters')
    params_list = os.listdir(params_path)
    single_exp_list = [x for x in params_list if '_multiday' not in x and 'params_' in x]
    mutliday_exp_list = [x for x in params_list if '_multiday' in x and 'params_' in x]

    if run_single_exp:
        for exp in single_exp_list:
            single_exp_time = time.time()
            config_filename = exp.split('.')[0]
            print(f'[STARTING] Single exp on {config_filename}!')
            os.system('python Run_single_exp.py --params_file ' + os.path.join('parameters',exp))
            print(f'[COMPLETED] {config_filename} in','{0:.2f}'.format(time.time()-single_exp_time), 's')
        print('[COMPLETED] Single experiment!')
    if run_multiday_exp:
        for multi_exp in mutliday_exp_list:
            multiday_exp_time = time.time()
            multiday_config_filename = multi_exp.split('.')[0]
            print(f'[STARTING] Multyday exp on {multiday_config_filename}!')
            os.system('python Run_multiday_exp.py --params_file ' + os.path.join('parameters',multi_exp))
            print(f'[COMPLETED] {multiday_config_filename} in','{0:.2f}'.format(time.time()-multiday_exp_time), 's')
        print('[COMPLETED] Multiday experiment!')
        
    

if __name__ == '__main__':
    main()