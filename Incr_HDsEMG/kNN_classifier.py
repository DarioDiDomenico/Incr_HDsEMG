import os
import Data_Handler
import numpy as np
from river import metrics
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

def accuracy(x):
    acc = np.empty((1,len(Data_Handler.label_map)))
    acc[:] = np.NaN
    for t_class in x.data.keys():
        if int(sum(x.data[t_class].values())) != 0:
            acc[0][int(t_class)] = x.data[t_class][t_class]/sum(x.data[t_class].values())
        else:
            acc[0][int(t_class)] = np.NaN
    return acc

def write_to_file_kNN(file_name, n_neighbors, algorithm, kf_results, save_folder):
    results_path = os.path.join(save_folder,*file_name.split(os.sep)[1:-1],'kNN')
    if not os.path.exists(results_path):
        os.makedirs(results_path)
        # Program to append to text file using write() function
    with  open(os.path.join(results_path,'kNN_val_batch_kf.txt'), "a") as file:
        file.write('n_neigh: '+ str(n_neighbors))
        file.write('\t')
        file.write('kf-acc: '+ str(np.around(kf_results['mean_acc'],3)))
        file.write('\t')
        file.write('Mean_kf-acc: '+ str(np.around(np.nanmean(kf_results['mean_acc']),5)))
        file.write('\t')
        file.write('STD_kf-acc: '+ str(np.around(np.nanstd(kf_results['mean_acc']),5)))
        file.write('\t')
        file.write('\n')
        file.close()

def main(n_neighbors, algorithm, file_name, parameters):

    # define dataset and data_loader    
    train_data, _ = Data_Handler.create_dataloader_test(file_name=file_name, parameters=parameters)
    
    nFeatures = train_data.shape[1]-1
    nCnt = int(((Data_Handler.sampling_rate*Data_Handler.rep_duration)-(parameters['window']-parameters['increment']))/parameters['increment'])
    train_data_resh = Data_Handler.dataset_reashape(train_data, nCnt)
    X_tr = train_data_resh[:,:nFeatures]
    Y_tr = train_data_resh[:,-1]
                         
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
        
        # Define model pre-training batch
        model=None
        model = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=algorithm)
        model.fit(X_tr,Y_tr)
        
        val_dict = {'cm': {},
                'accuracy': {}}
        val_dict['cm'] = metrics.ConfusionMatrix()

        # Validation batch to select the best hyperparameter lambda
        for x_val,y_val_true in zip(X_val,Y_val):
            y_pred_val = int(model.predict([x_val]))
            val_dict['cm'].update(y_val_true,y_pred_val)
        val_dict['accuracy'] = accuracy(val_dict['cm'])
        # print('Validation set cm:\n',cm_all_val)
        # print('Mean accuracy on Validation Batch: ' + str(np.mean(val_dict['accuracy'])))

        kf_results['cm'].append(val_dict['cm'])
        kf_results['mean_acc'].append(np.mean(val_dict['accuracy']))

    
    ## Write to txt file the results of kNN:
    if parameters['save_results']:
        write_to_file_kNN(file_name, n_neighbors, algorithm, kf_results, save_folder=parameters['save_folder'])
    
if __name__ == '__main__':
    main()