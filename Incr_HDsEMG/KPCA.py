from sklearn.decomposition import KernelPCA
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import time
from matplotlib.colors import ListedColormap

from Data_Handler import load_EMG_data, compute_save_pickle

def create_dataloader(file_name, window, increment):
    try:
        norm_data = load_EMG_data(file_name, filter_data=True)
    except FileNotFoundError:
        norm_data = compute_save_pickle(file_name,window,increment, filter_data = True)
    return norm_data

def kpca_analysis(data, labels, saving_path, transformer):
        
    kpca = transformer.transform(data)

    plt.figure()
    
    # Set Seaborn color palette to a colorblind-friendly palette
    sns.set_palette("bright")
    colorblind_colors = sns.color_palette("bright", as_cmap=True)

    plt.scatter(kpca[:, 0], kpca[:, 1], c=labels, cmap=ListedColormap(colorblind_colors), edgecolors='k', alpha=0.8)
    
    # Plot limits for limb difference subject
    # plt.xlim([-0.23, 0.62])  # Set x-axis limits limb difference subj
    # plt.ylim([-0.14, 0.26])  # Set y-axis limits limb difference subj

    # Plot limits for healthy subject
    plt.xlim([-0.51, 0.69])  # Set x-axis limits healthy subj
    plt.ylim([-0.43, 0.54])  # Set y-axis limits healthy subj

    if '_all_days.' in saving_path:
        plt.tick_params(axis='x', which='both', labelbottom=True, labelsize=20)
        plt.tick_params(axis='y', which='both', labelleft=True, labelsize=20)
    else:
        plt.tick_params(axis='x', which='both', labelbottom=False)
        plt.tick_params(axis='y', which='both', labelleft=False)

    plt.savefig(saving_path, dpi=50,transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close()

def write_to_file(saving_path,n_iter,kl):
    # Program to append to text file using write() function
    with  open(os.path.join('d:\\',*saving_path.split(os.sep)[1:-1],'KPCA.txt'), "a") as file:
        file.write('Subj:: '+ str(saving_path.split(os.sep)[4]))
        file.write('\t')
        file.write('Date: '+ str(saving_path.split(os.sep)[5]))
        file.write('\t')
        file.write('n_iter: '+ str(n_iter))
        file.write('\t')
        file.write('kl_divergence: '+ str(kl))
        file.write('\n')
        file.close()

def main():
    window = 400
    increment = 100
    data_path = 'DELTA' # Folder name for the DELTA dataset
    saving_folder_name = 'Results_KPCA' # New folder name where to save the KPCA figures
    gammas=[5e-2] # S03: 5e-2, S07: 1e-2
    # gammas=[1e-3,5e-3,1e-2,2e-2,3e-2,4e-2,5e-2,6e-2,7e-2,8e-2,9e-2,1,10] # Guess values for gamma
    subjects = ['P03'] # Add here subjects to analyze
    
    for sub in subjects:
        start_time = time.time()
        subj_path = os.path.join(data_path,sub)
        acq_days = ['0','1','2','3','4','5']              

        for gamma in gammas:
            data_all_days = []
            labels_all_days = []   
            # Definition of the KPCA model
            transformer = KernelPCA(n_components=2, kernel='rbf', gamma = gamma)

            print('Appending single days')
            for day in acq_days:
                # This for loop aims to append data to build KPCA model on the whole data of each subject
                print(f'Appending: {sub} {day}')
                file_name = os.path.join(subj_path, day, sub+'_'+'{:02d}'.format(int(day))+'.h5')
                norm_data = create_dataloader(file_name, window, increment)
                data = norm_data['features']['rms']['val']
                labels = norm_data['data']['labels']
                if np.isnan(data).any():
                    continue
                data_all_days.append(data)
                labels_all_days.append(labels)
            data_all_days = np.concatenate(data_all_days)
            labels_all_days = np.concatenate(labels_all_days)
            try:
                transformer.fit(data_all_days)
            except RuntimeError as e:
                print(e)    

            print('Working on single days')
            for day in acq_days:
                print(f'Processing: {sub} {day}')
                file_name = os.path.join(subj_path, day, sub+'_'+'{:02d}'.format(int(day))+'.h5')
                norm_data = create_dataloader(file_name, window, increment)
                data = norm_data['features']['rms']['val']
                labels = norm_data['data']['labels']
                if np.isnan(data).any():
                    continue    

                saving_folder = os.path.join(os.path.dirname(__file__),saving_folder_name+'_'+str(gamma))
                if not os.path.exists(saving_folder):
                    os.makedirs(saving_folder)

                saving_path = os.path.join(saving_folder, file_name.split(os.sep)[-3]+'_'+file_name.split(os.sep)[-2]+'_KPCA.png')

                kpca_analysis(data, labels, saving_path, transformer)
            
            saving_path = os.path.join(os.path.dirname(__file__),saving_folder_name+'_'+str(gamma),'KPCA_'+sub+'_all_days.png')
            
            kpca_analysis(data_all_days, labels_all_days, saving_path, transformer)
        print(f'\t\t\tTime needed for {sub}:',time.time()-start_time)
    
if __name__ == '__main__':
    main()
