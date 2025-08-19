import IL_classifier, LDA_classifier, kNN_classifier
import numpy as np
import time
import yaml
import os
from datetime import datetime
from Data_Handler import read_params_file, hyperpar_selection
start_time = time.time()

def write_log(file_name, exp_time, parameters):
    results_path = os.path.join(parameters['save_folder'],*file_name.split(os.sep)[1:-1],parameters['ML_model'])
    if not os.path.exists(results_path):
        os.makedirs(results_path)
        # Program to append to text file using write() function
    with  open(os.path.join(results_path,'logs_single_exp.txt'), "a") as file:
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
    select_algo = parameters['ML_model']
    for file_name in parameters['file_name']:
        single_exp_time = time.time()
        
        if select_algo =='RLSC':
            RF = parameters['RF']
            min_lamb = parameters['min_lamb'] 
            max_lamb = parameters['max_lamb'] 
            n_lamb = parameters['n_lamb'] 
            lambda_test = np.logspace(min_lamb,max_lamb,n_lamb)
            gamma = parameters['gamma'] # It won't be used in the linear case
            # gamma_test = np.logspace(np.log10(0.0005),np.log10(50),21)
            # n_components_test = [20, 40, 80, 160, 320, 380, 440, 500, 560, 620, 680, 740, 800, 860, 920, 980, 1040, 1100, 1160, 1220, 1280]
            # random_state_test = range(1,101)
            for lamb in lambda_test:
                print("-------------------------------------")
                print("Regularized Least Squares Classifier:")
                print('############STARTING############')
                print('lamb: ', lamb)
                eval(parameters['ML_script']+'.main(lamb, gamma, file_name, parameters)')

        if select_algo =='RFRLSC':
            RF = parameters['RF']
            min_lamb = parameters['min_lamb'] 
            max_lamb = parameters['max_lamb'] 
            n_lamb = parameters['n_lamb'] 
            min_gamma = parameters['min_gamma']  
            max_gamma = parameters['max_gamma']
            n_gamma = parameters['n_gamma']
            lambda_test = np.logspace(min_lamb,max_lamb,n_lamb)
            gamma_test = np.logspace(np.log10(min_gamma),np.log10(max_gamma),n_gamma)
            n_components_test = np.logspace(np.log10(parameters['min_n_components']),np.log10(parameters['max_n_components']),parameters['num_n_components']).astype(int)
            #best_n_components_test = [2779] #Needed to compute the best kernel optimization
            random_state_test = range(parameters['min_random_state'],parameters['max_random_state'])
            if parameters['lambda_gamma_opt']:
                for lamb in lambda_test:
                    for gamma in gamma_test:
                        print("------------------------------------------------------")
                        print("Random Features - Regularized Least Squares Classifier")
                        print('############STARTING############')
                        print('lamb: ', lamb, 'gamma', gamma)
                        eval(parameters['ML_script']+'.main(lamb, gamma, file_name, parameters)')
            else:
                for n_components in n_components_test:
                    for rand_state in random_state_test:
                        results_file_name = os.path.join(parameters['save_folder'],file_name.split(os.sep)[1],file_name.split(os.sep)[2] ,parameters['ML_model'],parameters['ML_model']+'_val_batch_kf.txt')
                        best_lamb, best_gamma = hyperpar_selection(file_name = results_file_name, robust_opt = parameters['robust_opt'], model=parameters['ML_model'])
                        print("------------------------------------------------------")
                        print("Random Features - Regularized Least Squares Classifier")
                        print('############STARTING############')
                        print('n_components: ', n_components, 'rand_state: ', rand_state )
                        eval(parameters['ML_script']+'.main(best_lamb, best_gamma, file_name, parameters, n_components, rand_state)')
       
        if select_algo =='kNN':
            min_val = parameters['min_n_neighbors']
            max_val = parameters['max_n_neighbors']
            n_val = parameters['num_n_neighbors']
            n_neighbors = np.linspace(min_val,max_val,n_val).astype(int) #TODO: Define them as odd-values -> 1 to 19 passo 2
            algorithm = parameters['algorithm']
            for neigh in n_neighbors:
                print("------------------------------")
                print("k-Nearest Neighbors Classifier")
                print('############STARTING############')
                print('neighbors: ', neigh, 'algorithm: ', algorithm)
                eval(parameters['ML_script']+'.main(neigh, algorithm, file_name, parameters)')
        
        print("The single exp on", *file_name.split(os.sep)[1:-1],' took ', "{0:.2f}".format(time.time()-single_exp_time), "s")
        if parameters['save_results']:
            write_log(file_name, time.time()-single_exp_time, parameters)
    
if __name__ == '__main__':
    opt = read_params_file()
    f = open(opt.params_file,'rb')
    parameters = yaml.load(f, Loader=yaml.FullLoader)
    main(parameters)
    print("The whole program took", "{0:.2f}".format(time.time()-start_time), "s")


