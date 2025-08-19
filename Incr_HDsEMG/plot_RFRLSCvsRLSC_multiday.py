import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ast
import os
import sys
from getopt import getopt

def batch_average(acc_batch, n_days):
    _tmp = []
    _items = acc_batch.size - acc_batch.shape[0]*10
    for i in acc_batch:
        _tmp.append(i.take(range(10,n_days*10)))
    # mean_acc = np.concatenate(_tmp).reshape(int(_items/10),10).mean(axis=0)
    mean_acc = np.mean(np.concatenate(_tmp).reshape(int(_items/10),10), axis=0)
    std_acc = np.concatenate(_tmp).reshape(int(_items/10),10).std(axis=0)
    acc_day1 = acc_batch.mean(axis=0).take(range(10))
    std_day1 = acc_batch.std(axis=0).take(range(10))
    final_acc = []
    final_std_acc = []
    final_acc.append(acc_day1)
    final_std_acc.append(std_day1)
    for i in range(n_days-1):
        final_acc.append(mean_acc)
        final_std_acc.append(std_acc)
    
    return np.concatenate(final_acc).flatten(), np.concatenate(final_std_acc).flatten()

def batch_quantile(acc_batch, n_days, quantiles):
    _tmp = []
    _tmp_acc_batch = []
    _items = acc_batch.size - acc_batch.shape[0]*10
    for i in acc_batch:
        _tmp.append(i.take(range(10,n_days*10)))
    quantile_acc = np.quantile(np.array(np.concatenate(_tmp).reshape(int(_items/10),10)), quantiles, axis=0)
    quantile_day1 = [i.take(range(10)) for i in np.quantile(np.array(np.concatenate(acc_batch)), quantiles, axis=0)]
    final_quantile = []
    final_quantile = np.concatenate((quantile_day1, quantile_acc), axis=1)
    for i in range(n_days-2):
        final_quantile = np.concatenate((final_quantile, quantile_acc), axis=1)

    return final_quantile

def load_results(file_name):
    perm_str = []
    mean_acc = []
    # Load raw data from file
    with open(file_name) as f:
        lines = f.readlines()
    for count,test in enumerate(lines):
        lines[count] = test.split('\t')
        # if len(lines[count]) == 5:
        perm_str.append(lines[count][2].replace('permutation: ',''))
        acc = lines[count][3].replace('mean_acc: ','')
        mean_acc.append(ast.literal_eval(acc.replace('nan', 'None')))
    return mean_acc

def main(subj_list, boxPlots=False, results_fold=''):
    save_folder = results_fold[0]
    n_reps = 10
    for idx, sub in enumerate(subj_list):
        file_name_RFRLSC_incr = os.path.join(save_folder, sub, 'multiday', 'RFRLSC', 'incr_permutation_results.txt')
        file_name_RLSC_incr = os.path.join(save_folder, sub, 'multiday', 'RLSC', 'incr_permutation_results.txt')
        # Import data.

        acc_RFRLSC_incr = load_results(file_name_RFRLSC_incr)
        acc_RFRLSC_incr = np.matrix(acc_RFRLSC_incr, dtype = np.float64)
        
        acc_RLSC_incr = load_results(file_name_RLSC_incr)
        acc_RLSC_incr =np.matrix(acc_RLSC_incr, dtype = np.float64)

        n_days = None
        if acc_RFRLSC_incr.shape[1] == acc_RLSC_incr.shape[1]:
            n_days = int(acc_RFRLSC_incr.shape[1]/n_reps)
        else:
            raise ValueError('Results have different number of days')
        
        if idx == 0:
            overall_acc_RFRLSC_incr = acc_RFRLSC_incr
            overall_acc_RLSC_incr = acc_RLSC_incr
        else:
            overall_acc_RFRLSC_incr = np.concatenate((overall_acc_RFRLSC_incr, acc_RFRLSC_incr),0)
            overall_acc_RLSC_incr = np.concatenate((overall_acc_RLSC_incr, acc_RLSC_incr),0)
    
    if not boxPlots:
        overall_diff=overall_acc_RFRLSC_incr-overall_acc_RLSC_incr
        mean_diff_incr = np.squeeze(np.array(overall_diff.mean(axis=0)))*100
        std_diff_incr = np.squeeze(np.array(overall_diff.std(axis=0)))*100

        quantile_diff_incr = np.quantile(np.array(overall_diff),[0.25,0.5,0.75],axis=0)*100

        x_ax = np.linspace(0,mean_diff_incr.shape[0]-1,num=mean_diff_incr.shape[0]).astype(int)
        tick_labels = []

        fig, ax = plt.subplots()
        # plt.setp(ax.get_xticklabels(), rotation = 45, ha="right",
        #         rotation_mode="anchor")

        for day in range(n_days):
            up_mean_std = mean_diff_incr[0+day*n_reps:10+day*n_reps]+std_diff_incr[0+day*n_reps:10+day*n_reps]
            low_mean_std = mean_diff_incr[0+day*n_reps:10+day*n_reps]-std_diff_incr[0+day*n_reps:10+day*n_reps]
            plt.plot(x_ax[0+day*n_reps:10+day*n_reps]-0.15, mean_diff_incr[0+day*n_reps:10+day*n_reps], color=sns.color_palette('Set2')[1])
            plt.vlines(x_ax[0+day*n_reps:10+day*n_reps]-0.15, low_mean_std, up_mean_std, color=sns.color_palette('Set2')[1])
            plt.axvline(x = 9+day*10, color = '#C5C9C7', linestyle=':')
            tick_labels.append([i+1+10*day for i in range(10)])
        if len(subj_list)==1:
            plt.title(f"Acc_RF-RLSC - Acc_RLSC on {n_days} days for "+file_name_RFRLSC_incr.split(os.sep)[1])
        else:
            plt.title("Median accuracy and quartiles on all days' permutations for all subjects")

        
        plt.legend(loc='lower left')
        plt.xlabel('Repetitions')
        plt.ylabel('Accuracy [%]')
        tick_labels = list(map(str, [item for sublist in tick_labels for item in sublist]))
        _tmp = ['\n\n'] * (len(tick_labels) * 2)
        _tmp[0::2] = tick_labels

        tick_labels = _tmp

        tick_locations = np.arange(n_days*10)

        new_labels = [ ''.join(x) for x in zip(tick_labels[0::2], tick_labels[1::2]) ]
        
        plt.xticks(tick_locations, new_labels)
        plt.tight_layout()
        plt.show()
        print('Stop')

    else:
        overall_diff=(overall_acc_RFRLSC_incr-overall_acc_RLSC_incr)*100
        Output = [np.nanmean(overall_diff[:,i:i+n_reps],axis=1) for i in range(0,overall_diff.shape[1] - n_reps + 1, n_reps)]
        box_data = np.concatenate(Output,axis=1)

        fig, ax = plt.subplots()
        ymin = -20.1
        ymax = 20.1
        ax.set_ylim([ymin, ymax])
        boxes_sep = 0.4
        plt.grid()
        sns.violinplot(data=box_data, color=".22", width=boxes_sep, ax=ax)
        # sns.stripplot(data=box_data, color=".22", ax=ax)
        sns.boxplot(data=box_data, color="whitesmoke", width=boxes_sep, ax=ax)
        plt.setp(ax.collections, alpha=.25)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.xlabel('Days')
        plt.ylabel('Accuracy [%]')
        # if len(subj_list)==1:
        #     plt.title(f"Acc_RF-RLSC - Acc_RLSC on {n_days} days for "+file_name_RFRLSC_incr.split(os.sep)[1])
        # else:
        #     subjects = ', '.join(subj_list)
        #     plt.title(f"Acc_RF-RLSC - Acc_RLSC on {subjects}")
        plt.yticks(range(-20,21,10))
        plt.tight_layout()
        ax.set_axisbelow(True)
        plt.show()

    
if __name__ == '__main__':
    try:
        opts, args = getopt(sys.argv[1:],'m:f:',['subj=','boxPlots=','results_fold='])
    except:
        print("Wrong command inputs")

    for option, argument in opts:
        if option == '--subj':
            subj_list = argument.replace(' ','').strip('][').split(',')
        if option == '--boxPlots':
            boxPlots_flag = argument.replace(' ','').strip('][').split(',')
            if boxPlots_flag[0] == 'True':
                boxPlots = True
            else:
                boxPlots = False
        if option == '--results_fold':
            results_fold = argument.replace(' ','').strip('][').split(',')

    main(subj_list,boxPlots,results_fold)