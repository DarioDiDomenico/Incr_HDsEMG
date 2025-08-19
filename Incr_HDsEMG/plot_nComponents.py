import matplotlib.pyplot as plt
import numpy as np
import os
import yaml
from Data_Handler import read_params_file

def load_data(file_name):
    n_components = []
    rnd_state = []
    mean_acc = []
    print('LOADING...')
    data_path = os.path.join(os.path.dirname(__file__), file_name)
    # Load raw data from file
    with open(file_name) as f:
        lines = f.readlines()
    for count,test in enumerate(lines):
        lines[count] = test.split('\t')
        if len(lines[count]) == 5: # Check if the 
            n_components.append(float(lines[count][0].replace('n_components: ','')))
            rnd_state.append(float(lines[count][1].replace('random_state: ','')))
            mean_acc.append(float(lines[count][3].replace('Mean_accuracy: ','')))
    print('DONE!')
    return n_components, rnd_state, mean_acc

def plot_accVSnCcomponents(n_components_test, averaged_mean_acc, std_mean_acc, exactKernel_mean_acc=None):
    fig, ax = plt.subplots()
    plt.setp(ax.get_xticklabels(), rotation = 45, ha="right",
            rotation_mode="anchor")
  
    plt.xticks(n_components_test)
    if exactKernel_mean_acc is not None:
        plt.hlines(exactKernel_mean_acc, np.min(n_components_test), np.max(n_components_test),color='red', linestyle='dashed')
    color = 'green'
    plt.plot(n_components_test, averaged_mean_acc,color='#008000')
    plt.fill_between(n_components_test, averaged_mean_acc-2*std_mean_acc, averaged_mean_acc+2*std_mean_acc, alpha=0.3, facecolor='#7EFF99',linewidth=2)
    plt.errorbar(n_components_test, averaged_mean_acc, 2*std_mean_acc, ecolor = color, linestyle='None', marker='^', markerfacecolor = color,markeredgecolor = color)
    #plt.plot([n_components_test[np.argmax(averaged_mean_acc)]], [np.max(averaged_mean_acc)], '-rD')
    plt.xlabel("# components")
    plt.ylabel("Accuracy [%]")
    ax.set_title("Accuracy of RFRLSC on the ValBatchSet")
    fig.tight_layout()
    #plt.show()
    
    print("Plot acc VS n_componenets!")

def violinplot(data,n_components_test,exactKernel_mean_acc=None):
    # Violinplot implementation
    ## combine these different collections into a list
    data_to_plot = data.T

    # Create a figure instance
    fig, ax = plt.subplots(figsize=(1.736, 1.5))
    ymin = 0.5
    ymax = 1.01
    ax.set_ylim([ymin, ymax])
    plt.grid()
    plt.setp(ax.get_xticklabels(), rotation = 45, ha="right", rotation_mode="anchor")
    plt.setp(ax, xticks=[y + 1 for y in range(data_to_plot.shape[1])], xticklabels=[str(int(comp))for comp in n_components_test])
    # Create the boxplot
    x_ax_min,x_ax_max = ax.get_xlim()
    if exactKernel_mean_acc is not None:
        plt.hlines(exactKernel_mean_acc, x_ax_min, x_ax_max,color='red', linestyle='dashed')
    ax.violinplot(data_to_plot, showmeans=True)
    plt.xlabel("# components")
    plt.ylabel("Accuracy [%]")
    fig.tight_layout()
    #plt.show()

    print('Plot shown!')

def main():
    opt = read_params_file()
    f = open(opt.params_file,'rb')
    parameters = yaml.load(f, Loader=yaml.FullLoader)
    select_algo = parameters['ML_model']
    for file_name in parameters['file_name']:
        results_path = os.path.join(parameters['save_folder'],*file_name.split(os.sep)[1:-1],parameters['ML_model'],'_'.join([parameters['ML_model'],'val_batch_nComponents.txt']))
    
        results_path_exactKernel = os.path.join(parameters['save_folder'],*file_name.split(os.sep)[1:-1],parameters['ML_model'],'BEST_'+'_'.join([parameters['ML_model'],'val_batch_nComponents.txt']))

        # Import data
        nComponents, random_state, mean_acc = load_data(results_path)
        exactKernel_nComponents, exactKernel_random_state, exactKernel_mean_acc = load_data(results_path_exactKernel)
        nComponents = np.array(nComponents)
        exactKernel_nComponents = np.array(exactKernel_nComponents)
        random_state = np.array(random_state)
        n_components_test = np.unique(nComponents)
        mean_acc = np.array(mean_acc)
        exactKernel_mean_acc = np.array(exactKernel_mean_acc)
        mean_acc_4each_rndstate = mean_acc.reshape(-1,len(np.unique(random_state)))
        averaged_mean_acc = np.average(mean_acc_4each_rndstate,axis=1)
        std_mean_acc = np.std(mean_acc_4each_rndstate,axis=1)

        violinplot(mean_acc_4each_rndstate, n_components_test, np.mean(exactKernel_mean_acc))
        if parameters['save_fig']:
                plt.savefig(os.path.join(os.sep.join(results_path.split(os.sep)[:-1]),'_'.join([select_algo,results_path.split(os.sep)[2]])+'_nComponents.png'))
if __name__ == '__main__':
    main()


