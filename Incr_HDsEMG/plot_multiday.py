import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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

def main(subj_list, boxPlots=False,results_fold=''):
    save_folder = results_fold[0]
    n_reps = 10
    for idx, sub in enumerate(subj_list):
        file_name_RFRLSC_batch = os.path.join(save_folder, sub, 'multiday', 'RFRLSC', 'batch_permutation_results.txt')
        file_name_RFRLSC_incr = os.path.join(save_folder, sub, 'multiday', 'RFRLSC', 'incr_permutation_results.txt')
        file_name_LDA_batch = os.path.join(save_folder, sub, 'multiday', 'LDA', 'batch_permutation_results.txt')
        file_name_kNN_batch = os.path.join(save_folder, sub, 'multiday', 'kNN', 'batch_permutation_results.txt')
        file_name_RLSC_batch = os.path.join(save_folder, sub, 'multiday', 'RLSC', 'batch_permutation_results.txt')
        file_name_RLSC_incr = os.path.join(save_folder, sub, 'multiday', 'RLSC', 'incr_permutation_results.txt')
        plot_mean = False # True: plot the mean and std - False: plot the median and quartiles
        # Import data.

        acc_RFRLSC_batch = load_results(file_name_RFRLSC_batch)
        acc_RFRLSC_incr = load_results(file_name_RFRLSC_incr)
        acc_RFRLSC_batch =np.matrix(acc_RFRLSC_batch, dtype = np.float64)
        acc_RFRLSC_incr = np.matrix(acc_RFRLSC_incr, dtype = np.float64)
        
        acc_LDA_batch = load_results(file_name_LDA_batch)
        acc_kNN_batch = load_results(file_name_kNN_batch)
        acc_RLSC_batch = load_results(file_name_RLSC_batch)
        acc_RLSC_incr = load_results(file_name_RLSC_incr)
        acc_LDA_batch =np.matrix(acc_LDA_batch, dtype = np.float64)
        acc_kNN_batch =np.matrix(acc_kNN_batch, dtype = np.float64)
        acc_RLSC_batch =np.matrix(acc_RLSC_batch, dtype = np.float64)
        acc_RLSC_incr =np.matrix(acc_RLSC_incr, dtype = np.float64)

        n_days = None
        if acc_RLSC_batch.shape[1] == acc_RFRLSC_batch.shape[1] == acc_kNN_batch.shape[1] == acc_LDA_batch.shape[1]:
            n_days = int(acc_RLSC_batch.shape[1]/n_reps)
        else:
            raise ValueError('Results have different number of days')
        
        if idx == 0:
            overall_acc_RFRLSC_batch = acc_RFRLSC_batch
            overall_acc_RFRLSC_incr = acc_RFRLSC_incr
            overall_acc_LDA_batch = acc_LDA_batch
            overall_acc_kNN_batch = acc_kNN_batch
            overall_acc_RLSC_batch = acc_RLSC_batch
            overall_acc_RLSC_incr = acc_RLSC_incr
        else:
            overall_acc_RFRLSC_batch = np.concatenate((overall_acc_RFRLSC_batch, acc_RFRLSC_batch),0)
            overall_acc_RFRLSC_incr = np.concatenate((overall_acc_RFRLSC_incr, acc_RFRLSC_incr),0)
            overall_acc_LDA_batch = np.concatenate((overall_acc_LDA_batch, acc_LDA_batch),0)
            overall_acc_kNN_batch = np.concatenate((overall_acc_kNN_batch, acc_kNN_batch),0)
            overall_acc_RLSC_batch = np.concatenate((overall_acc_RLSC_batch, acc_RLSC_batch),0)
            overall_acc_RLSC_incr = np.concatenate((overall_acc_RLSC_incr, acc_RLSC_incr),0)
    
    if not boxPlots:  
        mean_acc_RFRLSC_batch, std_acc_RFRLSC_batch = batch_average(overall_acc_RFRLSC_batch, n_days)
        mean_acc_RLSC_batch, std_acc_RLSC_batch = batch_average(overall_acc_RLSC_batch, n_days)
        mean_acc_kNN_batch, std_acc_kNN_batch = batch_average(overall_acc_kNN_batch, n_days)
        mean_acc_LDA_batch, std_acc_LDA_batch = batch_average(overall_acc_LDA_batch, n_days)
        mean_acc_RFRLSC_incr = np.squeeze(np.array(overall_acc_RFRLSC_incr.mean(axis=0)))
        std_acc_RFRLSC_incr = np.squeeze(np.array(overall_acc_RFRLSC_incr.std(axis=0)))
        mean_acc_RLSC_incr = np.squeeze(np.array(overall_acc_RLSC_incr.mean(axis=0)))
        std_acc_RLSC_incr = np.squeeze(np.array(overall_acc_RLSC_incr.std(axis=0)))

        quantile_acc_RFRLSC_batch = batch_quantile(overall_acc_RFRLSC_batch, n_days, [0.25,0.5,0.75])
        quantile_acc_LDA_batch = batch_quantile(overall_acc_LDA_batch, n_days, [0.25,0.5,0.75])
        quantile_acc_kNN_batch = batch_quantile(overall_acc_kNN_batch, n_days, [0.25,0.5,0.75])
        quantile_acc_RLSC_batch =batch_quantile(overall_acc_RLSC_batch, n_days, [0.25,0.5,0.75])
        quantile_acc_RFRLSC_incr = np.quantile(np.array(overall_acc_RFRLSC_incr),[0.25,0.5,0.75],axis=0)
        quantile_acc_RLSC_incr = np.quantile(np.array(overall_acc_RLSC_incr),[0.25,0.5,0.75],axis=0)

        x_ax = np.linspace(0,mean_acc_RFRLSC_batch.shape[1]-1,num=mean_acc_RFRLSC_batch.shape[1])
        tick_labels = []

        fig, ax = plt.subplots()
        # plt.setp(ax.get_xticklabels(), rotation = 45, ha="right",
        #         rotation_mode="anchor")
        if plot_mean:
            plt.vlines(x_ax+0.05, mean_acc_RFRLSC_batch-std_acc_RFRLSC_batch , mean_acc_RFRLSC_batch+std_acc_RFRLSC_batch, color=sns.color_palette('Set2')[0], label='RF-RLSC batch')
            plt.vlines(x_ax-0.15, mean_acc_RFRLSC_incr-std_acc_RFRLSC_incr , mean_acc_RFRLSC_incr+std_acc_RFRLSC_incr, color=sns.color_palette('Set2')[1], label='RF-RLSC incremental')
            plt.vlines(x_ax+0.15, mean_acc_LDA_batch-std_acc_LDA_batch , mean_acc_LDA_batch+std_acc_LDA_batch, color=sns.color_palette('Set2')[2], label='LDA batch')
            plt.vlines(x_ax-0.05, mean_acc_kNN_batch-std_acc_kNN_batch , mean_acc_kNN_batch+std_acc_kNN_batch, color=sns.color_palette('Set2')[3], label='kNN batch')
            plt.vlines(x_ax+0.25, mean_acc_RLSC_batch-std_acc_RLSC_batch , mean_acc_RLSC_batch+std_acc_RLSC_batch, color=sns.color_palette('Set2')[4], label='RLSC batch')
            plt.vlines(x_ax-0.25, mean_acc_RLSC_incr-std_acc_RLSC_incr , mean_acc_RLSC_incr+std_acc_RLSC_incr, color=sns.color_palette('Set2')[5], label='RLSC incremental')
            for day in range(n_days):
                plt.plot(x_ax[0+day*n_reps:10+day*n_reps]+0.05, mean_acc_RFRLSC_batch.tolist()[0][0+day*n_reps:10+day*n_reps], color=sns.color_palette('Set2')[0])
                plt.plot(x_ax[0+day*n_reps:10+day*n_reps]-0.15, mean_acc_RFRLSC_incr[0+day*n_reps:10+day*n_reps], color=sns.color_palette('Set2')[1])
                plt.plot(x_ax[0+day*n_reps:10+day*n_reps]+0.15, mean_acc_LDA_batch.tolist()[0][0+day*n_reps:10+day*n_reps], color=sns.color_palette('Set2')[2])
                plt.plot(x_ax[0+day*n_reps:10+day*n_reps]-0.05, mean_acc_kNN_batch.tolist()[0][0+day*n_reps:10+day*n_reps], color=sns.color_palette('Set2')[3])
                plt.plot(x_ax[0+day*n_reps:10+day*n_reps]+0.25, mean_acc_RLSC_batch.tolist()[0][0+day*n_reps:10+day*n_reps], color=sns.color_palette('Set2')[4])
                plt.plot(x_ax[0+day*n_reps:10+day*n_reps]-0.25, mean_acc_RLSC_incr[0+day*n_reps:10+day*n_reps], color=sns.color_palette('Set2')[5])
                plt.axvline(x = 9+day*10, color = '#C5C9C7', linestyle=':')
                tick_labels.append([i+1+10*day for i in range(10)])
            if len(subj_list)==1:
                plt.title(f"Accuracy averaged on permutations of {n_days} days for "+file_name_RFRLSC_batch.split(os.sep)[1])
            else:
                plt.title("Median accuracy and quartiles on all days' permutations for all subjects")
        else:
            plt.vlines(x_ax+0.05, quantile_acc_RFRLSC_batch[0,:],quantile_acc_RFRLSC_batch[2,:], color=sns.color_palette('Set2')[0], label='RF-RLSC batch')
            plt.plot(x_ax+0.05, mean_acc_RFRLSC_batch.tolist()[0], '^',color=sns.color_palette('Set2')[0], markersize=5)
            plt.vlines(x_ax-0.15, quantile_acc_RFRLSC_incr[0,:],quantile_acc_RFRLSC_incr[2,:], color=sns.color_palette('Set2')[1], label='RF-RLSC incremental')
            plt.plot(x_ax-0.15, mean_acc_RFRLSC_incr, '^', color=sns.color_palette('Set2')[1], markersize=5)
            plt.vlines(x_ax+0.15, quantile_acc_LDA_batch[0,:],quantile_acc_LDA_batch[2,:], color=sns.color_palette('Set2')[2], label='LDA batch')
            plt.plot(x_ax+0.15, mean_acc_LDA_batch.tolist()[0], '^', color=sns.color_palette('Set2')[2], markersize=5)
            plt.vlines(x_ax-0.05, quantile_acc_kNN_batch[0,:],quantile_acc_kNN_batch[2,:], color=sns.color_palette('Set2')[3], label='kNN batch')
            plt.plot(x_ax-0.05, mean_acc_kNN_batch.tolist()[0], '^', color=sns.color_palette('Set2')[3], markersize=5)
            plt.vlines(x_ax+0.25, quantile_acc_RLSC_batch[0,:],quantile_acc_RLSC_batch[2,:], color=sns.color_palette('Set2')[4], label='RLSC batch')
            plt.plot(x_ax+0.25, mean_acc_RLSC_batch.tolist()[0], '^', color=sns.color_palette('Set2')[4], markersize=5)
            plt.vlines(x_ax-0.25, quantile_acc_RLSC_incr[0,:],quantile_acc_RLSC_incr[2,:], color=sns.color_palette('Set2')[5], label='RLSC incremental')
            plt.plot(x_ax-0.25, mean_acc_RLSC_incr, '^', color=sns.color_palette('Set2')[5], markersize=5)
            for day in range(n_days):
                plt.plot(x_ax[0+day*n_reps:10+day*n_reps]+0.05, quantile_acc_RFRLSC_batch[1,0+day*n_reps:10+day*n_reps], color=sns.color_palette('Set2')[0])
                plt.plot(x_ax[0+day*n_reps:10+day*n_reps]-0.15, quantile_acc_RFRLSC_incr[1,0+day*n_reps:10+day*n_reps], color=sns.color_palette('Set2')[1])
                plt.plot(x_ax[0+day*n_reps:10+day*n_reps]+0.15, quantile_acc_LDA_batch[1,0+day*n_reps:10+day*n_reps], color=sns.color_palette('Set2')[2])
                plt.plot(x_ax[0+day*n_reps:10+day*n_reps]-0.05, quantile_acc_kNN_batch[1,0+day*n_reps:10+day*n_reps], color=sns.color_palette('Set2')[3])
                plt.plot(x_ax[0+day*n_reps:10+day*n_reps]+0.25, quantile_acc_RLSC_batch[1,0+day*n_reps:10+day*n_reps], color=sns.color_palette('Set2')[4])
                plt.plot(x_ax[0+day*n_reps:10+day*n_reps]-0.25, quantile_acc_RLSC_incr[1,0+day*n_reps:10+day*n_reps], color=sns.color_palette('Set2')[5])
                plt.axvline(x = 9+day*10, color = '#C5C9C7', linestyle=':')
                tick_labels.append([i+1+10*day for i in range(10)])
            if len(subj_list)==1:
                plt.title("Median accuracy and quartiles on all days' permutations for "+file_name_RFRLSC_batch.split(os.sep)[1])
            else:
                plt.title("Median accuracy and quartiles on all days' permutations for all subjects")
        
        ymin = 0.0
        ymax = 1.01
        ax.set_ylim([ymin, ymax])
        plt.legend(loc='lower left')
        plt.xlabel('Repetitions')
        plt.ylabel('Accuracy')
        tick_labels = list(map(str, [item for sublist in tick_labels for item in sublist]))
        _tmp = ['\n\n'] * (len(tick_labels) * 2)
        _tmp[0::2] = tick_labels

        tick_labels = _tmp

        tick_locations = np.arange(n_days*10)

        new_labels = [ ''.join(x) for x in zip(tick_labels[0::2], tick_labels[1::2]) ]
        
        plt.xticks(tick_locations, new_labels)
        plt.yticks(np.around(np.arange(0,1.1,0.1),1).tolist())
        plt.tight_layout()
        plt.show()
    
    else:

        output_RFRLSC_batch = np.concatenate([np.nanmean(overall_acc_RFRLSC_batch[:,i:i+n_reps],axis=1)*100 for i in range(0,overall_acc_RFRLSC_batch.shape[1] - n_reps + 1, n_reps)],axis=1)
        output_RFRLSC_incr = np.concatenate([np.nanmean(overall_acc_RFRLSC_incr[:,i:i+n_reps],axis=1)*100 for i in range(0,overall_acc_RFRLSC_incr.shape[1] - n_reps + 1, n_reps)],axis=1)        
        output_LDA_batch = np.concatenate([np.nanmean(overall_acc_LDA_batch[:,i:i+n_reps],axis=1)*100 for i in range(0,overall_acc_LDA_batch.shape[1] - n_reps + 1, n_reps)],axis=1)
        output_kNN_batch = np.concatenate([np.nanmean(overall_acc_kNN_batch[:,i:i+n_reps],axis=1)*100 for i in range(0,overall_acc_kNN_batch.shape[1] - n_reps + 1, n_reps)],axis=1)
        output_RLSC_batch = np.concatenate([np.nanmean(overall_acc_RLSC_batch[:,i:i+n_reps],axis=1)*100 for i in range(0,overall_acc_RLSC_batch.shape[1] - n_reps + 1, n_reps)],axis=1)
        output_RLSC_incr = np.concatenate([np.nanmean(overall_acc_RLSC_incr[:,i:i+n_reps],axis=1)*100 for i in range(0,overall_acc_RLSC_incr.shape[1] - n_reps + 1, n_reps)],axis=1)

        pd_RFRLSC_batch = pd.DataFrame(output_RFRLSC_batch.tolist(), columns=list(range(0,6))).assign(Algorithm='RFRLSC batch')
        pd_RFRLSC_incr = pd.DataFrame(output_RFRLSC_incr.tolist(), columns=list(range(0,6))).assign(Algorithm='RFRLSC incr')
        pd_LDA_batch = pd.DataFrame(output_LDA_batch.tolist(), columns=list(range(0,6))).assign(Algorithm='LDA batch')
        pd_kNN_batch = pd.DataFrame(output_kNN_batch.tolist(), columns=list(range(0,6))).assign(Algorithm='kNN batch')
        pd_RLSC_batch = pd.DataFrame(output_RLSC_batch.tolist(), columns=list(range(0,6))).assign(Algorithm='RLSC batch')
        pd_RLSC_incr = pd.DataFrame(output_RLSC_incr.tolist(), columns=list(range(0,6))).assign(Algorithm='RLSC incr')

        cdf = pd.concat([pd_RFRLSC_batch,pd_RFRLSC_incr,pd_LDA_batch,pd_kNN_batch,pd_RLSC_batch,pd_RLSC_incr])
        mdf = pd.melt(cdf, id_vars=['Algorithm'], var_name=['Day'])    # MELT
        mdf = mdf.sort_values('Algorithm')
        fig, ax = plt.subplots()
        ymin = 0.0
        ymax = 100.1
        ax.set_ylim([ymin, ymax])
        plt.grid()
        palette_tmp = 'Blues'

        ax = sns.boxplot(x="Algorithm", y="value", hue="Day", palette=palette_tmp, data=mdf)  # RUN PLOT 
        sns.despine(offset=10, trim=True)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_ylabel('Accuracy [%]')
        ax.set_axisbelow(True)
        plt.yticks(range(0,101,10))
        plt.tight_layout()
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