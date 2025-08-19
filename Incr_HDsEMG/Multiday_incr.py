import os
import math
import scipy
import Data_Handler
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from river import metrics
from RLS_classifier import rls_classifier

def accuracy(x):
    acc = np.empty((1,len(Data_Handler.label_map)))
    acc[:] = np.NaN
    for t_class in x.data.keys():
        if int(sum(x.data[t_class].values())) != 0:
            acc[0][int(t_class)] = x.data[t_class][t_class]/sum(x.data[t_class].values())
        else:
            acc[0][int(t_class)] = np.NaN
    return acc

def Euclidean_distance(X,Y):
    D_matrix = scipy.spatial.distance_matrix(X, Y)
    eucl_dist_arr = np.triu(D_matrix).flatten()
    nonzero_mask = np.nonzero(eucl_dist_arr)
    eucl_dist_arr = eucl_dist_arr[nonzero_mask]

    print("Median ",np.median(eucl_dist_arr))
    print("Max ",np.max(eucl_dist_arr))
    print("Min ",np.min(eucl_dist_arr))

def write_to_file(file_name, perm_str, overall_mean_acc, parameters, predict_time, save_folder):
    results_path = os.path.join(save_folder,file_name.split(os.sep)[1],'multiday',parameters['ML_model'])
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    # Program to append to text file using write() function
    with  open(os.path.join(results_path,'incr_permutation_results.txt'), "a") as file:
        file.write('best_lamb: '+ str(round(parameters['lambda'], 5)))
        file.write('\t')
        if parameters['ML_model'] == 'RFRLSC':
            file.write('best_gamma: '+ str(round(parameters['gamma'], 5)))
            file.write('\t')
        elif parameters['ML_model'] == 'RLSC':
            file.write('best_gamma: '+ str(None))
            file.write('\t')
        file.write('permutation: '+ perm_str)
        file.write('\t')
        file.write('mean_acc: '+ str([round(item,5) for item in overall_mean_acc]))
        file.write('\t')
        file.write('mean_predict_time: '+ str(np.mean(predict_time)))
        file.write('\t')
        file.write('std_predict_time: '+ str(np.std(predict_time)))
        file.write('\t')
        file.write('\n')
        file.close() 

def main(parameters, perm, n_components=500, random_state=1):
    parameters['n_components'] = n_components
    parameters['random_state'] = random_state
    predict_time = []
    n_days = len(perm)
    results_file_name = os.path.join(parameters['save_folder'],*perm[0].split(os.sep)[1:-1],parameters['ML_model'],parameters['ML_model']+'_val_batch_kf.txt')
    if parameters['ML_model'] == 'RFRLSC':
        best_lamb, best_gamma = Data_Handler.hyperpar_selection(file_name = results_file_name, robust_opt = parameters['robust_opt'], model=parameters['ML_model'])
        print('best_lamb: ', best_lamb,'\nbest_gamma: ', best_gamma)
        parameters['gamma'] = best_gamma
        parameters['lambda'] = best_lamb
    elif parameters['ML_model'] == 'RLSC':
        best_lamb = Data_Handler.hyperpar_selection(file_name = results_file_name, robust_opt = parameters['robust_opt'], model=parameters['ML_model'])
        print('best_lamb: ', best_lamb)
        parameters['lambda'] = best_lamb
    perm_str_list = []
    [perm_str_list.append(x.split(os.sep)[2]) for x in perm]
    perm_str = '&&'.join(perm_str_list)

    for n_day,current_day in enumerate(perm):
        train_data, test_data = Data_Handler.create_dataloader_test(file_name=current_day, parameters=parameters)
        
        nFeatures = train_data.shape[1]-1
        count = 1
        nCount_flag = False
        hist_cnt = 0
        nCnt = int(((Data_Handler.sampling_rate*Data_Handler.rep_duration)-(parameters['window']-parameters['increment']))/parameters['increment'])
        overall_mean_acc_test = []
        overall_mean_acc_test2 = []

        test_data = Data_Handler.dataset_reashape(test_data, nCnt)
        X_tr = train_data[:,:nFeatures]
        Y_tr = train_data[:,-1]
        
        print("-----------------------------")
        if parameters['RF']:
            print("Random Features - Regularized Least Squares Classifier:")
        else:
            print("Regularized Least Squares Classifier:")
        if n_day == 0: # Pre-training batch on the first day
            model = rls_classifier(best_lamb,nFeatures,X_tr,Y_tr)
        else:
            ## Incremental update of the model
            for x_train_incr,y_train_incr in zip(X_tr,Y_tr):
                model.learn_one(x_train_incr, y_train_incr)  

        cm = metrics.ConfusionMatrix()
        hist = {'cm': {},
                'accuracy': {},
                'mean_acc': {}}
        for f in range(0,math.ceil(len(test_data)/(nCnt*len(Data_Handler.label_map)))):
            hist['cm']['cm_' + str(f)]     = metrics.ConfusionMatrix()

        X_te = test_data[:,:nFeatures]
        Y_te = test_data[:,-1]

        for x_upd,y_upd in zip(X_te,Y_te):
            
            _time_before = time.process_time()
            y_pred = model.predict_one(x_upd)
            predict_time.append(time.process_time()-_time_before)
            cm.update(y_upd, y_pred)
            hist['cm']['cm_'+str(hist_cnt)].update(y_upd, y_pred)
        
            if (((nCount_flag == 1) & ((count % (nCnt*len(Data_Handler.label_map))) == 0)) | (count == len(X_te))):
                hist['accuracy']['acc_'+str(hist_cnt)] = accuracy(hist['cm']['cm_'+str(hist_cnt)])
                hist['mean_acc']['mean_acc_'+str(hist_cnt)] = np.mean(hist['accuracy']['acc_'+str(hist_cnt)])
                print("ConfMat hist day", n_day+1,":\n", hist['cm']['cm_'+str(hist_cnt)])
                print("Mean accuracy hist day", n_day+1,":\n", hist['mean_acc']['mean_acc_'+str(hist_cnt)])
                hist_cnt += 1
                print('-----------------------------------------')    
            count += 1
            nCount_flag = True
        
        print('Mean accuracy on test1 of day ', n_day+1,': ' + str(np.mean(hist['accuracy']['acc_'+str(hist_cnt-1)])))

        overall_mean_acc_test = Data_Handler.update_overall_results(hist)

        if n_day == 0:
            overall_mean_acc = []
        overall_mean_acc.append([np.nan])
        overall_mean_acc.append([np.nan])
        overall_mean_acc.append(overall_mean_acc_test)
    overall_mean_acc = list(np.concatenate(overall_mean_acc).flat)

    ######################################################################
    ## Write the overall results to file
    if parameters['save_results']:
        write_to_file(perm[0], perm_str, overall_mean_acc, parameters, predict_time, save_folder=parameters['save_folder'])

    x_ax = np.linspace(0,len(overall_mean_acc)-1,num=len(overall_mean_acc))
    fig, ax = plt.subplots()
    ymin = 0.0
    ymax = 1.01
    ax.set_ylim([ymin, ymax])
    ax.set_xlim([-1, len(overall_mean_acc)+1])
    plt.setp(ax.get_xticklabels(), rotation = 45, ha="right",
            rotation_mode="anchor")
    
    ax.axvspan(-0.5, 1.5, alpha=0.1, color='red', label = 'Pre-training')
    for day in range(n_days):
        start_ind = 1 + day * 10
        end_ind = 10 + day * 10
        if day != 0:
            ax.axvspan(start_ind-1.5, start_ind-0.5, alpha=0.1, color='green')
        plt.plot(x_ax[start_ind:end_ind],overall_mean_acc[start_ind:end_ind],marker='o',color=sns.color_palette('Set2')[day], label='Day '+str(day+1))
        
    plt.legend(ncol = 2)
    plt.xlabel('Repetitions')
    plt.ylabel('Accuracy')
    if parameters['RF']:
        plt.title('RFRLSC model trained on the 20% of Day1 data')
        if parameters['save_fig']:
            results_path = os.path.join(parameters['save_folder'],perm[0].split(os.sep)[1],'multiday',parameters['ML_model'])
            if not os.path.exists(results_path):
                os.makedirs(results_path)
            plt.savefig(os.path.join(results_path,'+'.join(perm_str_list)+'_incr.png'))
    else:
        plt.title('RLSC model trained on the 20% of Day1 data')
        if parameters['save_fig']:
            results_path = os.path.join(parameters['save_folder'],perm[0].split(os.sep)[1],'multiday',parameters['ML_model'])
            if not os.path.exists(results_path):
                os.makedirs(results_path)
            plt.savefig(os.path.join(results_path,'+'.join(perm_str_list)+'_incr.png'))
    # plt.show()
    plt.close()

if __name__ == '__main__':
    main()
