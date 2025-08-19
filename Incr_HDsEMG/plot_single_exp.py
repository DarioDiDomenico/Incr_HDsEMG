import matplotlib.pyplot as plt
import numpy as np
import os
import yaml
from mpl_toolkits.axes_grid1 import make_axes_locatable
from Data_Handler import read_params_file, load_RFRLSC_results, hyperpar_selection, load_RLSC_results, load_kNN_results

def plot_contourf(c_gammas, r_lambdas, best_lamb, best_gamma, z_acc, std_acc, N, cmap, robust_opt):
    
    origin = 'lower'
    fig, ax = plt.subplots(figsize=(8, 6))
    if robust_opt:
        CS3 = ax.contourf(z_acc-2*std_acc, N, cmap=cmap, origin = None, extent = [0,len(c_gammas)-1,0,len(r_lambdas)-1], vmin = -0.2, vmax = 1)
    else:
        CS3 = ax.contourf(z_acc, N, cmap=cmap, origin = origin, vmin = -0.2, vmax = 1)
    ax.set_xlabel("gamma \u03B3")
    ax.set_ylabel("lambda \u03BB")

    ind_r_lambdas = np.where(r_lambdas==best_lamb)
    ind_c_gammas = np.where(c_gammas==best_gamma)
    plt.scatter(ind_c_gammas, ind_r_lambdas, c="blue", marker = '*', label='opt hyperpar')

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(c_gammas)), labels=c_gammas)
    ax.set_yticks(np.arange(len(r_lambdas)), labels=r_lambdas)
    ax.set_yticklabels(['{:.2e}'.format(x) for x in r_lambdas])
    ax.set_xticklabels(['{:.2e}'.format(x) for x in c_gammas])
    ax.invert_yaxis()

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    plt.colorbar(CS3, ax=ax, format='%0.2f')
    #plt.show()
    return

def plot_heatmap(c_gammas, r_lambdas, best_lamb, best_gamma, z_acc, std_acc, cmap, robust_opt = False):
    
    fig, ax = plt.subplots(figsize=(8, 6))
    if robust_opt:
        im = ax.imshow(z_acc-2*std_acc, cmap=cmap, vmin = -0.2, vmax = 1)
    else:
        im = ax.imshow(z_acc, cmap=cmap, vmin = -0.2, vmax = 1)
    
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(c_gammas)), labels=c_gammas)
    ax.set_yticks(np.arange(len(r_lambdas)), labels=r_lambdas)
    ax.set_yticklabels(['{:.2e}'.format(x) for x in r_lambdas])
    ax.set_xticklabels(['{:.2e}'.format(x) for x in c_gammas])

    ind_r_lambdas = np.where(r_lambdas==best_lamb)
    ind_c_gammas = np.where(c_gammas==best_gamma)
    plt.scatter(ind_c_gammas, ind_r_lambdas, c="blue", marker = '*', label='opt hyperpar')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    plt.xlabel("gamma \u03B3")
    plt.ylabel("lambda \u03BB")
    ax.set_title("k-fold validation accuracy of RFRLSC")
    fig.tight_layout()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)
    #plt.show()

    return

def load_LDA_results(file_name):
    n_componenets = []
    mean_acc = []
    std_acc = []
    # Load raw data from file
    with open(file_name) as f:
        lines = f.readlines()
    for count,test in enumerate(lines):
        lines[count] = test.split('\t')
        if len(lines[count]) == 5: # Check file's structure 
            n_componenets.append(float(lines[count][0].replace('n_components: ','')))
            mean_acc.append(float(lines[count][2].replace('Mean_kf-acc: ','')))
            std_acc.append(float(lines[count][3].replace('STD_kf-acc: ','')))
    return n_componenets, np.array(mean_acc), np.array(std_acc)

def main():

    opt = read_params_file()
    f = open(opt.params_file,'rb')
    parameters = yaml.load(f, Loader=yaml.FullLoader)
    select_algo = parameters['ML_model']
    for file_name in parameters['file_name']:
        results_path = os.path.join(parameters['save_folder'],*file_name.split(os.sep)[1:-1],parameters['ML_model'],'_'.join([parameters['ML_model'],'val_batch_kf.txt']))
        robust_opt = parameters['robust_opt']
        # Import data.
        if select_algo == 'RLSC':
            opt_comp, mean_acc, std_acc = load_RLSC_results(results_path)
        elif select_algo == 'LDA':
            opt_comp, mean_acc, std_acc = load_LDA_results(results_path)
        elif select_algo == 'kNN':
            opt_comp, mean_acc, std_acc = load_kNN_results(results_path)
        elif select_algo == 'RFRLSC':
            opt_comp, mean_acc, std_acc = load_RFRLSC_results(results_path)
            lambdas = opt_comp[0]
            gammas = opt_comp[1]
            ind_max = np.where(mean_acc==max(mean_acc))
            max_mean_acc = max(mean_acc)
            cmap = 'turbo'
            N = 100
            print('lambda ',lambdas[ind_max[0][0]])
            print('gamma ',gammas[ind_max[0][0]])
            print('max_mean_acc ',max_mean_acc)
            r_lambdas = np.unique(lambdas)
            c_gammas = np.unique(gammas)
            z_acc = mean_acc.reshape(len(r_lambdas),len(c_gammas))
            std_acc = std_acc.reshape(len(r_lambdas),len(c_gammas))

        if select_algo == 'RFRLSC':
            best_lamb, best_gamma = hyperpar_selection(results_path, robust_opt = robust_opt, model = parameters['ML_model'])
            print('Best Hyperparams:\nlambda: ', best_lamb,'gamma: ', best_gamma)
            plot_contourf(c_gammas, r_lambdas, best_lamb, best_gamma, z_acc, std_acc, N, cmap, robust_opt)
            if parameters['save_fig']:
                plt.savefig(os.path.join(os.sep.join(results_path.split(os.sep)[:-1]),'_'.join([select_algo,results_path.split(os.sep)[2]])+'_contour.png'))
            plot_heatmap(c_gammas, r_lambdas, best_lamb, best_gamma, z_acc, std_acc, cmap, robust_opt = robust_opt)
            if parameters['save_fig']:
                plt.savefig(os.path.join(os.sep.join(results_path.split(os.sep)[:-1]),'_'.join([select_algo,results_path.split(os.sep)[2]])+'_heatmap.png'))
    
        else:
            best_hyp = hyperpar_selection(results_path, robust_opt = robust_opt, model = parameters['ML_model'])
            r_opt_comp = np.unique(opt_comp)
            fig, ax = plt.subplots()
            # Rotate the tick labels and set their alignment.
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                    rotation_mode="anchor")
            plt.xticks(r_opt_comp)
            plt.errorbar(opt_comp, mean_acc, std_acc, color='#008000')
            if robust_opt:
                _tmp = mean_acc-2*std_acc
            else:
                _tmp = mean_acc
            if select_algo == 'RLSC':
                ax.set_xscale('log')
                plt.plot(best_hyp, np.max(_tmp), marker = "h", color = 'r', label = 'Best hyperpar', zorder = 10)
                plt.xlabel("lambda \u03BB")
                plt.grid()
                ax.set_title("Accuracy of RLSC on the Validation Batch: " + file_name.split(os.sep)[2])
            elif select_algo == 'LDA':
                plt.xlabel("n_components")
                plt.plot(best_hyp, np.max(_tmp), marker = "h", color = 'r', label = 'Best hyperpar', zorder = 10)
                ax.set_title("Accuracy of LDA on the Validation Batch: " + file_name.split(os.sep)[2])
            elif select_algo == 'kNN':
                plt.xlabel("n_neigh")
                plt.plot(best_hyp, np.max(_tmp), marker = "h", color = 'r', label = 'Best hyperpar', zorder = 10)
                ax.set_title("Accuracy of kNN on the Validation Batch: " + file_name.split(os.sep)[2])
            
            ymin = 0.5
            ymax = 1.01
            ax.set_ylim([ymin, ymax])
            plt.ylabel("Accuracy")
            plt.legend(loc='lower left')
            fig.tight_layout()
            #plt.show()
            if parameters['save_fig']:
                plt.savefig(os.path.join(os.sep.join(results_path.split(os.sep)[:-1]),'_'.join([select_algo,results_path.split(os.sep)[2]])+'.png'))



if __name__ == '__main__':
    main()


