import argparse
import os
import math
import pickle
import h5py
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.kernel_approximation import RBFSampler
from scipy.signal import iirnotch, filtfilt

def read_params_file():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params_file", default='No params file selected!', type=str, help='Parameters file')
    opt = parser.parse_args()
    return opt

# Meaning of labels
label_map = {0: 'Resting',
             1: 'Hand Closing',
             2: 'Hand Opening',
             3: 'Wrist Pronation',
             4: 'Wrist Supination',
             5: 'Wrist Flexion',
             6: 'Wrist Extension'}

sampling_rate = 2000    # Sampling frequency of the acquisition device
rep_duration = 2        # Seconds of acquisition for each repetition
notch_freq = 50

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, axis=0, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered_data = filtfilt(b, a, data, axis=axis)
    return filtered_data

def notch_filter(data, freqs, fs, axis=0):
    b, a = iirnotch(freqs, Q=10, fs=fs)
    filtered_data = filtfilt(b, a, data, axis=axis)
    return filtered_data

def preprocessing(data, window = 400, increment = 100, nEMGChan=64):
    ms = []
    rms = []
    labels = []

    features_idx = []
    for j in range(int(data.shape[0]/(rep_duration*sampling_rate))):
        _data = data[rep_duration*sampling_rate*j:(j+1)*rep_duration*sampling_rate,:]
        _, indices, counts = np.unique(_data[:,-1], return_index=True, return_counts=True)
        for i, c in zip(indices, counts):
            idx_test = int(i + c)
            features_idx = (slice(i, idx_test))
            class_data = _data[features_idx]
            sliding_win_view = np.lib.stride_tricks.sliding_window_view(class_data, window, axis=0)
            sliding_win_labels = sliding_win_view[:, -1]
            labels.append(sliding_win_labels[::increment][:,-1])
            sliding_win_view = sliding_win_view[:,:nEMGChan]
            # Mean Square (MSQ)
            ms.append(np.sum(np.square(sliding_win_view[::increment]), axis=-1) / window)

            # Root Mean Square
            rms.append(np.sqrt(np.sum(np.square(sliding_win_view[::increment]), axis=-1) / window))

    features_dict = {
        'data': {'data': data,
                'labels': np.concatenate(labels)},
        'features': {'ms':{'val': np.concatenate(ms), 'max': [], 'min': []},
                     'rms':{'val': np.concatenate(rms), 'max': [], 'min': []}},         
        'window': window,
        'increment': increment
    }

    return features_dict

def normalization(features_dict):

    min_len = min([len(x['val']) for x in features_dict['features'].values()])

    for feat in features_dict['features']: 
        
        feature = features_dict['features'][feat]['val']
        
        feature = feature[:min_len]
        max_val = np.max(feature, axis=0)
        min_val = np.min(feature, axis=0)
        features_dict['features'][feat]['max'] = max_val
        features_dict['features'][feat]['min'] = min_val
        
        features_dict['features'][feat]['val'] = (feature - min_val) / (max_val - min_val)

    return features_dict

def load_data(file_name):
    with h5py.File(file_name, 'r') as file:
        data = file[file_name.split(os.sep)[1]+'_'+'{:02d}'.format(int(file_name.split(os.sep)[2]))][:]
    return data

def compute_save_pickle(file_name, window, increment, filter_data = True):
    data = load_data(file_name)
    split_name = file_name.split(os.sep)[:-1]
    # Prepocessing of data
    if filter_data:
        filtered_signal = []

        acq_sample_len = int(rep_duration*sampling_rate)
        for j in range(int(data.shape[0]/acq_sample_len)):
            # Apply the notch filter to remove interference at the specified frequency
            emg_signal_filtered = notch_filter(data[acq_sample_len*j:(j+1)*acq_sample_len,:64], freqs=50, fs=sampling_rate, axis=0)

            # Apply the bandpass filter to the EMG signal
            filtered_signal.append(butter_bandpass_filter(emg_signal_filtered, lowcut=20, highcut=500, fs=sampling_rate, axis=0))

        data[:,:64] = np.row_stack(filtered_signal)

    proc_data = preprocessing(data, window, increment)
    norm_data = normalization(proc_data)

    # # To fasten the experiments the data has been loaded and the norm_data has been computed
    # # Computation of norm_data.p
    if filter_data:
        with open(os.path.join(*split_name,split_name[1]+'_'+split_name[2]+'_filtered_pickle.p'), 'wb') as f:
            pickle.dump(norm_data, f)
    else:
        with open(os.path.join(*split_name,split_name[1]+'_'+split_name[2]+'_pickle.p'), 'wb') as f:
            pickle.dump(norm_data, f)
    
    return norm_data

def load_EMG_data(file_name, filter_data):
    split_name = file_name.split(os.sep)[:-1]
    #Loading of norm_data.p
    if filter_data:
        with open(os.path.join(*split_name,split_name[1]+'_'+split_name[2]+'_filtered_pickle.p'), 'rb') as fp:
            norm_data = pickle.load(fp)
    else:
        with open(os.path.join(*split_name,split_name[1]+'_'+split_name[2]+'_pickle.p'), 'rb') as fp:
            norm_data = pickle.load(fp)
    return norm_data

def create_dataloader_test(file_name, parameters):
    try:
        norm_data = load_EMG_data(file_name, parameters['filter_emg_data'])
    except FileNotFoundError:
        norm_data = compute_save_pickle(file_name,parameters['window'],parameters['increment'], filter_data = parameters['filter_emg_data'])
    
    # Splitting training and test set
    _, indices, counts = np.unique(norm_data['data']['labels'], return_index=True, return_counts=True)

    train_batch_ind = []
    test_ind        = []
    for i, c in zip(indices, counts):
        idx_batch       = int(i + c * 0.2)  # 20% of pre-training
        idx_test        = int(i + c)        # 80% of test set
        train_batch_ind.append(slice(i, idx_batch))
        test_ind.append(slice(idx_batch, idx_test))
    
    if parameters['RF']:
        # Concatenate features data
        rbf_feature = RBFSampler(gamma = parameters['gamma'], random_state = parameters['random_state'], n_components = parameters['n_components'])
        X_features = rbf_feature.fit_transform(norm_data['features']['rms']['val'])
        conc_data = np.append(X_features,norm_data['data']['labels'][:,None],axis = 1)
    else:
        # Concatenate rms data
        conc_data = np.append(norm_data['features']['rms']['val'],norm_data['data']['labels'][:,None],axis = 1)
    
    train_data = np.concatenate([conc_data[idx] for idx in train_batch_ind])
    test_data = np.concatenate([conc_data[idx] for idx in test_ind])

    return train_data, test_data

def create_dataloader_whole_test(file_name, parameters):
    try:
        norm_data = load_EMG_data(file_name, parameters['filter_emg_data'])
    except FileNotFoundError:
        norm_data = compute_save_pickle(file_name,parameters['window'],parameters['increment'], filter_data = parameters['filter_emg_data'])
    
    # Splitting training and test set
    _, indices, counts = np.unique(norm_data['data']['labels'], return_index=True, return_counts=True)

    train_batch_ind = []
    test_ind        = []
    for i, c in zip(indices, counts):
        idx_batch       = int(i + c * 0.0)  # 0% of pre-training
        idx_test        = int(i + c)        # 100% of test set
        train_batch_ind.append(slice(i, idx_batch))
        test_ind.append(slice(idx_batch, idx_test))
    
    if parameters['RF']:
        # Concatenate features data
        rbf_feature = RBFSampler(gamma = parameters['gamma'], random_state = parameters['random_state'], n_components = parameters['n_components'])
        X_features = rbf_feature.fit_transform(norm_data['features']['rms']['val'])
        conc_data = np.append(X_features,norm_data['data']['labels'][:,None],axis = 1)
    else:
        # Concatenate rms data
        conc_data = np.append(norm_data['features']['rms']['val'],norm_data['data']['labels'][:,None],axis = 1)
    
    train_data = np.concatenate([conc_data[idx] for idx in train_batch_ind])
    test_data = np.concatenate([conc_data[idx] for idx in test_ind])

    return train_data, test_data

def dataset_reashape(dataset, nCnt):
    reshaped_data = []
    classes, indices, counts = np.unique(dataset[:,-1], return_index=True, return_counts=True)

    batch_ind = {}
    for f in range(0,math.ceil(max(counts)/nCnt)):
        batch_ind['ind_' + str(f)]     = []
    for i, c in zip(indices, counts):
        index = np.where(indices == i)
        for n in range(0,int(np.ceil(c/nCnt))):
            idx_start = int(i + nCnt*(n))
            idx_end = int(i + nCnt*(n+1))
            if index[0].item()+1 < len(classes) and idx_end>indices[index[0].item()+1]:
                idx_end = indices[index[0].item()+1].item()
            batch_ind['ind_' + str(n)].append(slice(idx_start, idx_end))
            
    for idx in range(0,len(batch_ind.keys())):
        reshaped_data.append(np.concatenate([dataset[slc] for slc in batch_ind['ind_' + str(idx)]]))
    reshaped_data = np.concatenate(reshaped_data)
    
    return reshaped_data

def update_overall_results(hist):
    return list(hist['mean_acc'].values())

def hyperpar_selection(file_name, robust_opt = True, model=None):
    if model is None:
        raise ValueError('You must define the desired model for hyperparameters selection!')
    if model == 'RFRLSC':
        '''file_name must contain the path starting from the Results folder to the RFRLSC_valbatch_kf.txt'''
        opt, mean_acc, std_acc = load_RFRLSC_results(file_name)
        r_lambdas = np.unique(opt[0])
        c_gammas = np.unique(opt[1])
        z_acc = mean_acc.reshape(len(r_lambdas),len(c_gammas))
        z_std = std_acc.reshape(len(r_lambdas),len(c_gammas))
        if robust_opt:
            z = z_acc-2*z_std
        else:
            z = z_acc
        max_xy = np.squeeze(np.where(z == z.max()))
        if max_xy.ndim == 2:
            max_lamb_list = np.squeeze(np.where(max_xy[0] == max_xy[0].max()))
            if max_lamb_list.size > 1:
                max_lamb = max_xy[0][max_lamb_list[0]]
                max_gamma = max_xy[1][max_lamb_list[0]:].min()
            else:
                max_lamb = int(max_xy[0][max_lamb_list])
                max_gamma = int(max_xy[1][max_lamb_list])
        else:
            max_lamb = max_xy[0]
            max_gamma = max_xy[1]
        best_lamb = r_lambdas[max_lamb]
        best_gamma = c_gammas[max_gamma]

        return best_lamb, best_gamma
    
    elif model == 'RLSC':
        opt_comp, mean_acc, std_acc = load_RLSC_results(file_name)
        if robust_opt:
            _tmp = mean_acc-2*std_acc
        else:
            _tmp = mean_acc
        idx_max = np.squeeze(np.where(_tmp==np.max(_tmp)))
        if idx_max.size > 1:
            max_lamb = int(idx_max[-1])
        else:
            max_lamb = int(idx_max)
        
        best_lamb = opt_comp[max_lamb]
        return best_lamb

    elif model == 'kNN':
        opt_comp, mean_acc, std_acc = load_kNN_results(file_name)
        if robust_opt:
            _tmp = mean_acc-2*std_acc
        else:
            _tmp = mean_acc
        idx_max = np.squeeze(np.where(_tmp==np.max(_tmp)))
        if idx_max.size > 1:
            max_n_neigh = int(idx_max[-1])
        else:
            max_n_neigh = int(idx_max)
        
        best_n_neigh = opt_comp[max_n_neigh]
        return int(best_n_neigh)

def load_RFRLSC_results(file_name):
    lambdas = []
    gammas = []
    std = []
    mean_acc = []
    # Load raw data from file
    with open(file_name) as f:
        lines = f.readlines()
    for count,test in enumerate(lines):
        lines[count] = test.split('\t')
        if len(lines[count]) == 6:
            lambdas.append(float(lines[count][0].replace('lambda: ','')))
            gammas.append(float(lines[count][1].replace('gamma: ','')))
            mean_acc.append(float(lines[count][3].replace('Mean_kf-acc: ','')))
            std.append(float(lines[count][4].replace('STD_kf-acc: ','')))
    return [lambdas, gammas], np.array(mean_acc), np.array(std)

def load_RLSC_results(file_name):
    lambdas = []
    mean_acc = []
    std_acc = []
    # Load raw data from file
    with open(file_name) as f:
        lines = f.readlines()
    for count,test in enumerate(lines):
        lines[count] = test.split('\t')
        if len(lines[count]) == 5: # Check file's structure 
            lambdas.append(float(lines[count][0].replace('lambda: ','')))
            mean_acc.append(float(lines[count][2].replace('Mean_kf-acc: ','')))
            std_acc.append(float(lines[count][3].replace('STD_kf-acc: ','')))
    return lambdas, np.array(mean_acc), np.array(std_acc)

def load_kNN_results(file_name):
    n_neigh = []
    mean_acc = []
    std_acc = []
    # Load raw data from file
    with open(file_name) as f:
        lines = f.readlines()
    for count,test in enumerate(lines):
        lines[count] = test.split('\t')
        if len(lines[count]) == 5: # Check file's structure 
            n_neigh.append(float(lines[count][0].replace('n_neigh: ','')))
            mean_acc.append(float(lines[count][2].replace('Mean_kf-acc: ','')))
            std_acc.append(float(lines[count][3].replace('STD_kf-acc: ','')))
    return n_neigh, np.array(mean_acc), np.array(std_acc)

def batch_comb(file_name_list):
    comb_list = []
    for i in file_name_list:
        _n_list = []
        _n_list.append(i)
        [_n_list.append(x) for x in file_name_list if x != i]
        comb_list.append(_n_list)
    return comb_list