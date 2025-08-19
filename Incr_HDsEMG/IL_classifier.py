import os
import scipy
import Data_Handler
import numpy as np
from river import metrics
from RLS_classifier import rls_classifier
from sklearn.model_selection import KFold

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

def write_to_file_RFRLSC(file_name, lamb, gamma, kf_results, save_folder):
    # Program to append to text file using write() function
    results_path = os.path.join(save_folder,*file_name.split(os.sep)[1:-1],'RFRLSC')
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    with  open(os.path.join(results_path,'RFRLSC_val_batch_kf.txt'), "a") as file:
        file.write('lambda: '+ str(lamb))
        file.write('\t')
        file.write('gamma: '+ str(gamma))
        file.write('\t')
        file.write('kf-acc: '+ str(np.around(kf_results['mean_acc'],3)))
        file.write('\t')
        file.write('Mean_kf-acc: '+ str(np.around(np.mean(kf_results['mean_acc']),5)))
        file.write('\t')
        file.write('STD_kf-acc: '+ str(np.around(np.std(kf_results['mean_acc']),5)))
        file.write('\t')
        file.write('\n')
        file.close()

def write_to_file_RLSC(file_name, lamb, kf_results, save_folder):
    results_path = os.path.join(save_folder,*file_name.split(os.sep)[1:-1],'RLSC')
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    # Program to append to text file using write() function
    with  open(os.path.join(results_path,'RLSC_val_batch_kf.txt'), "a") as file:
        file.write('lambda: '+ str(lamb))
        file.write('\t')
        file.write('kf-acc: '+ str(np.around(kf_results['mean_acc'],3)))
        file.write('\t')
        file.write('Mean_kf-acc: '+ str(np.around(np.mean(kf_results['mean_acc']),5)))
        file.write('\t')
        file.write('STD_kf-acc: '+ str(np.around(np.std(kf_results['mean_acc']),5)))
        file.write('\t')
        file.write('\n')
        file.close() 

def write_to_file_features(file_name, n_components, random_state, val_dict, save_folder):
    results_path = os.path.join(save_folder,*file_name.split(os.sep)[1:-1],'RFRLSC')
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    # Program to append to text file using write() function
    with  open(os.path.join(results_path,'RFRLSC_val_batch_nComponents.txt'), "a") as file:
        file.write('n_components: '+ str(n_components))
        file.write('\t')
        file.write('random_state: '+ str(random_state))
        file.write('\t')
        file.write('accuracy: '+ str(np.around(val_dict['accuracy'],3)))
        file.write('\t')
        file.write('Mean_accuracy: '+ str(np.mean(val_dict['accuracy'])))
        file.write('\t')
        file.write('\n')
        file.close()

def main(lamb, gamma, file_name, parameters, n_components=500, random_state=1):
    parameters['gamma'] = gamma
    parameters['lambda'] = lamb
    parameters['n_components'] = n_components
    parameters['random_state'] = random_state
    train_data, _ = Data_Handler.create_dataloader_test(file_name=file_name, parameters=parameters)
    nFeatures = train_data.shape[1]-1
    
    # Compute the Euclidian distance to select a proper hyperparameter space
    # Euclidean_distance(X,X)

    n_splits = 10
    kf = KFold(n_splits = n_splits)

    balance_val = round(round(len(train_data[:,-1])/n_splits)/len(Data_Handler.label_map))
    train_data_resh = Data_Handler.dataset_reashape(train_data,balance_val)
    X = train_data_resh[:,:nFeatures]
    Y = train_data_resh[:,-1]

    kf_results = {'cm':[], 'mean_acc':[]}        
    
    for train_index, val_index in kf.split(X):
        X_tr, X_val = X[train_index], X[val_index]
        Y_tr, Y_val = Y[train_index], Y[val_index]

        # Pre-training batch
        model=None
        model = rls_classifier(parameters['lambda'],nFeatures,X_tr,Y_tr)

        val_dict = {'cm': {},
                'accuracy': {}}
        val_dict['cm'] = metrics.ConfusionMatrix()

        # Validation batch to select the best hyperparameter lambda
        for x_val,y_val_true in zip(X_val,Y_val):
            y_pred_val = model.predict_one(x_val)
            val_dict['cm'].update(y_val_true,y_pred_val)
        val_dict['accuracy'] = accuracy(val_dict['cm'])
        # print('Validation set cm:\n',cm_all_val)
        # print('Mean accuracy on Validation Batch: ' + str(np.mean(val_dict['accuracy'])))

        kf_results['cm'].append(val_dict['cm'])
        kf_results['mean_acc'].append(np.mean(val_dict['accuracy']))

    if parameters['RF']:
        if parameters['lambda_gamma_opt']:
            ## To compute the best lambda-gamma hyperparameters save the results on .txt file
            # # Write to txt file the results of RFRLSC
            if parameters['save_results']:
                if not os.path.exists(parameters['save_folder']):
                    os.makedirs(parameters['save_folder'])
                write_to_file_RFRLSC(file_name, parameters['lambda'], parameters['gamma'], kf_results, save_folder=parameters['save_folder'])
        else:
            ## To compute the best n_components-rand_state hyperparameters save the results on .txt file
            ## Write to txt file the results of RFRLSC
            if parameters['save_results']:
                if not os.path.exists(parameters['save_folder']):
                    os.makedirs(parameters['save_folder'])
                write_to_file_features(file_name, n_components, random_state, val_dict, save_folder=parameters['save_folder'])
    else:
        ## Write to txt file the results of RLSC
        if parameters['save_results']:
            if not os.path.exists(parameters['save_folder']):
                    os.makedirs(parameters['save_folder'])
            write_to_file_RLSC(file_name, parameters['lambda'], kf_results, save_folder=parameters['save_folder'])

if __name__ == '__main__':
    main()
