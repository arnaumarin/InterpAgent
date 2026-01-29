from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import random_split
import yaml

class Data_processor():
    
    waveform_pkl = "X_waveform.pkl"
    spike_id_pkl = "Y_spike_id.pkl"
    extremum_ch_pkl = "Y_spike_id_noise.pkl"
    
    #non_spike_id = [101,106,126,128]
    non_spike_id = []
    
    def __init__(self, dirs, day_id_str, qt_level=8):
        self.data_folder = dirs.dataset_path
        self.raw_path = dirs.pkl_path
        self.shank_positions = np.load(dirs.sp_path)
        self.train_data_folder = dirs.train_data_path
        self.test_data_folder = dirs.test_data_path
        self.params_path = dirs.params_path
        
        self.noise_date_folder = self.data_folder/'noise_data'
        self.label_data_folder = self.data_folder/'label_data'
        
        self.day_id_str = day_id_str
        self.n_day = len(day_id_str)
        self.qt_level = qt_level
        
        """with (open(path, "rb")) as openfile:
            spike_id_all = pickle.load(openfile)
            self.keep_label = np.unique(spike_id_all[~np.isin(spike_id_all, self.non_spike_id+[-1])])"""
        self.keep_label = np.load(dirs.ku_path)
        self.n_label = len(self.keep_label)
        self.label_index = {value: idx for idx, value in enumerate(self.keep_label)}
        print(f'#Label: {1+self.n_label}')
        
        self.processed_data = dict(zip(self.day_id_str,[None]*self.n_day))
        for day in day_id_str:
            self.processed_data[day] = self.process_data(day)
        
    def process_data(self, day_id):
        print(f'Processing {day_id}..',end=' ')
        
        path =   self.raw_path/day_id
        
        with (open(path/self.spike_id_pkl, "rb")) as openfile:
            spike_id_all = pickle.load(openfile)
        with (open(path/self.extremum_ch_pkl, "rb")) as openfile:
            extremum_ch = np.array(pickle.load(openfile))
        with (open(path/self.waveform_pkl, "rb")) as openfile:
            waveform_data = pickle.load(openfile)
        
        spike_id_all = np.array(spike_id_all)
        n_spike = spike_id_all.shape[0]
        is_spike = np.isin(spike_id_all, self.keep_label)
        custom_norm = Custom_Norm(self.shank_positions, extremum_ch)
        ##############################################################################################
        noise_gt = np.zeros((n_spike, ))
        noise_gt[is_spike] = 1
        ##############################################################################################
        label_gt = np.zeros((n_spike, ))
        label_gt[~is_spike] = -1
        label_gt[is_spike] = np.array([self.label_index[x] for x in spike_id_all[is_spike]])
        ##############################################################################################
        #pred_loc = location_cal_group(self.shank_positions, waveform_data, extremum_ch)[:,:-1]
        #self.proed_loc = pred_loc
        #pred_loc = custom_norm.minmax_normalize(pred_loc, feature_range=(0,1))
        #pred_loc = custom_norm.quantize(pred_loc, self.qt_level)
        pred_loc = self.shank_positions[:,:-1][extremum_ch]
        #pred_loc = custom_norm.minmax_normalize(pred_loc, feature_range=(0,1))
        #pred_loc = custom_norm.quantize(pred_loc, self.qt_level)
        ##############################################################################################
        #waveform_multi_ = custom_norm.sigmoid_normalize(waveform_data)
        #waveform_multi_ = custom_norm.quantize(waveform_multi_, self.qt_level)
        waveform_multi = custom_norm.group_shank(waveform_data)
        waveform_multi = waveform_multi.reshape(n_spike, -1)
        waveform_single = waveform_data[np.arange(n_spike), np.array(extremum_ch).astype('int'), :]
        
        print(f'Complete!')
        
        return noise_gt, label_gt, waveform_multi, waveform_single, pred_loc
    
    def gen_train_data(self, train_days:list, is_noise, val_ratio=0.8, small=None):
        if is_noise:
            print("Generating Noise detection train set ..")
        else:
            print("Generating Label classification train set ..")
        save_folder = self.noise_date_folder if is_noise else self.label_data_folder
        save_folder.mkdir(exist_ok=True, parents=True)
        x = []
        y = []
        aa = []
        for day_id in train_days:
            noise_gt, label_gt, waveform_multi, waveform_single, pred_loc = self.processed_data[day_id]
            aa += list(np.unique(label_gt))
            len_spike = noise_gt.shape[0]
            mask = [True]*len_spike if is_noise else noise_gt == 1

            multi = waveform_multi[mask]
            single = waveform_single[mask]
            loc = pred_loc[mask]
            x.append(np.concatenate((multi,single,loc), axis=1))
            if is_noise:
                y.append(noise_gt)
            else:
                y.append(label_gt[mask])
        print(f'# Label: {np.unique(aa).shape}')
        train_x = np.concatenate(x,axis=0)
        train_y = np.concatenate(y,axis=0)
        assert train_x.shape[0]==train_y.shape[0]
        
        train_folder = save_folder/'train_data'
        train_folder.mkdir(exist_ok=True)
        
        total_size = train_x.shape[0]
        train_size = int(total_size * val_ratio)
        val_size = train_x.shape[0] - train_size
        train_idx, val_idx = random_split(range(total_size), [train_size, val_size])
        train_x_ = train_x[list(train_idx)]
        train_y_ = train_y[list(train_idx)]
        val_x = train_x[list(val_idx)]
        val_y = train_y[list(val_idx)]
        print(f'Num Train set: {train_x_.shape[0]}    Num Validation set: {val_x.shape[0]}    Input size: {train_x_.shape[1]}    output size: {np.unique(train_y).shape[0]}\n')
        np.save(train_folder/'train_x', train_x_)
        np.save(train_folder/'train_y', train_y_)
        np.save(train_folder/'test_x', val_x)
        np.save(train_folder/'test_y', val_y)
            
    def gen_test_data(self, is_noise):
        if is_noise:
            print("Generating Noise detection test set")
        else:
            print("Generating Label classification test set")
        save_folder = self.noise_date_folder if is_noise else self.label_data_folder
        save_folder.mkdir(exist_ok=True, parents=True)
        test_folder = save_folder/'test_data'
        test_folder.mkdir(exist_ok=True)
        
        for day_id in self.day_id_str:
            day_folder = test_folder/day_id
            if not day_folder.exists(): day_folder.mkdir()
            noise_gt, label_gt, waveform_multi, waveform_single, pred_loc = self.processed_data[day_id]
            len_spike = noise_gt.shape[0]
            mask = [True]*len_spike if is_noise else noise_gt == 1

            multi = waveform_multi[mask]
            single = waveform_single[mask]
            loc = pred_loc[mask]
            test_x = np.concatenate((multi,single,loc), axis=1)
            if is_noise:
                test_y = noise_gt
            else:
                test_y = label_gt[mask]
            assert test_x.shape[0]==test_y.shape[0]
            np.save(day_folder/'test_x', test_x)
            np.save(day_folder/'test_y', test_y)
            print(f'{day_id} / num data: {test_x.shape[0]}\n')
            
    def gen_test_data_small(self, is_noise, n_per_label=50):
        if is_noise:
            print("Generating Noise detection small test set")
        else:
            print("Generating Label classification small test set")
        save_folder = self.noise_date_folder if is_noise else self.label_data_folder
        save_folder.mkdir(exist_ok=True, parents=True)
        test_folder = save_folder/'test_data_small'
        test_folder.mkdir(exist_ok=True)
        for day_id in self.day_id_str:
            day_folder = test_folder/day_id
            if not day_folder.exists(): day_folder.mkdir()
            noise_gt, label_gt, waveform_multi, waveform_single, pred_loc = self.processed_data[day_id]
            ip_shape = waveform_multi.shape[1] + waveform_single.shape[1] + pred_loc.shape[1]
            test_x = np.empty((0,ip_shape))
            test_y = np.empty((0,))
            for i in np.unique(label_gt):
                if i==-1: continue
                label_idx = np.where(label_gt == i)[0]
                n_sample = len(label_idx)//n_per_label
                if n_sample == 0:
                    print(f'{day_id} {int(i)}th label has only {len(label_idx)} spikes')
                    continue
                label_idx = label_idx[:n_sample]
                multi = waveform_multi[label_idx]
                single = waveform_single[label_idx]
                loc = pred_loc[label_idx]
                x = np.concatenate((multi,single,loc), axis=1)
                if is_noise:
                    y = np.full((n_sample),1,dtype=int)
                else:
                    y = np.full((n_sample),i,dtype=int)
                test_x = np.concatenate((test_x,x), axis=0)
                test_y = np.concatenate((test_y,y), axis=0)
            else:
                if is_noise:
                    noise_idx = np.where(noise_gt == 0)[0]
                    n_sample = len(noise_idx)//n_per_label
                    noise_idx = noise_idx[:n_sample]
                    multi = waveform_multi[noise_idx]
                    single = waveform_single[noise_idx]
                    loc = pred_loc[noise_idx]
                    x = np.concatenate((multi,single,loc), axis=1)
                    y = np.full((n_sample),0,dtype=int)
                    test_x = np.concatenate((test_x,x), axis=0)
                    test_y = np.concatenate((test_y,y), axis=0)
                assert test_x.shape[0]==test_y.shape[0]
                
                np.save(day_folder/'test_x', test_x)
                np.save(day_folder/'test_y', test_y)
                print(f'{day_id} / num data: {test_x.shape[0]}\n')
                
    def gen_unified_train_data(self, train_days:list, val_ratio=0.8):
        print("Generating unified label classification train set ..")
        x = []
        y = []
        noise_id = self.n_label
        for day_id in train_days:
            noise_gt, label_gt, waveform_multi, waveform_single, pred_loc = self.processed_data[day_id]
            
            multi = waveform_multi
            single = waveform_single
            loc = pred_loc
            x.append(np.concatenate((multi,single,loc), axis=1))
            
            unified_label_gt = label_gt.copy()
            unified_label_gt[unified_label_gt == -1] = noise_id
            y.append(unified_label_gt)
        train_x = np.concatenate(x,axis=0)
        train_y = np.concatenate(y,axis=0)
        assert train_x.shape[0]==train_y.shape[0]
        
        train_folder = self.train_data_folder
        train_folder.mkdir(exist_ok=True, parents=True)
        
        total_size = train_x.shape[0]
        train_size = int(total_size * val_ratio)
        val_size = train_x.shape[0] - train_size
        train_idx, val_idx = random_split(range(total_size), [train_size, val_size])
        train_x_ = train_x[list(train_idx)]
        train_y_ = train_y[list(train_idx)]
        val_x = train_x[list(val_idx)]
        val_y = train_y[list(val_idx)]
        spike_ratio = (val_y != noise_id).sum()/len(val_y)
        
        self.output_size = np.unique(train_y).shape[0]
        self.input_size = train_x_.shape[1]
        print(f'Num Train set: {train_x_.shape[0]}    Num Validation set: {val_x.shape[0]}    Input size: {self.input_size}    output size: {self.output_size}   spike ratio: {spike_ratio:0.3f}\n')
        
        np.save(train_folder/'train_x', train_x_)
        np.save(train_folder/'train_y', train_y_)
        np.save(train_folder/'test_x', val_x)
        np.save(train_folder/'test_y', val_y)
        
    def gen_unified_test_data(self):
        print("Generating unified label classification test set ..")
        test_folder = self.test_data_folder
        test_folder.mkdir(exist_ok=True, parents=True)
        noise_id = self.n_label
        for day_id in self.day_id_str:
            day_folder = test_folder/day_id
            if not day_folder.exists(): day_folder.mkdir()
            noise_gt, label_gt, waveform_multi, waveform_single, pred_loc = self.processed_data[day_id]
            len_spike = noise_gt.shape[0]

            multi = waveform_multi
            single = waveform_single
            loc = pred_loc
            test_x = np.concatenate((multi,single,loc), axis=1)
            
            unified_label_gt = label_gt.copy()
            unified_label_gt[unified_label_gt == -1] = noise_id
            test_y = unified_label_gt
            assert test_x.shape[0]==test_y.shape[0]
            np.save(day_folder/'test_x', test_x)
            np.save(day_folder/'test_y', test_y)
            print(f'{day_id} / num data: {test_x.shape[0]}\n')
            
    def save_params(self, model_name = 'trained_model1',epochs=20,device='cpu'):
        params = {
            'model_name': model_name,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'epochs': epochs,
            'device': device
        }
        
        with open(self.params_path, 'w') as file:
            yaml.dump(params, file, default_flow_style=False)


class Custom_Norm():
    def __init__(self, shank_position, channel_id):
        self.shank_position = shank_position
        self.channel_id = channel_id
        
    def sigmoid_normalize(self, data):
        mean_vals = np.mean(data, axis=(0, 2), keepdims=True)
        norm_data = data - mean_vals
        norm_data /= 100
        norm_data = 1 / (1 + np.exp(-norm_data))
        return norm_data
    
    def group_shank(self, data):
        z_values_to_match = self.shank_position[self.channel_id, 2]
        matching_indices = [np.where(self.shank_position[:, 2] == z_val)[0] for z_val in z_values_to_match]
        group_data = data[np.arange(data.shape[0])[:,None], matching_indices,:]
        return group_data

    def minmax_normalize(self,data,feature_range):
        scaler = StandardScaler()
        min_max_scaler = MinMaxScaler(feature_range=feature_range)
        norm_data = scaler.fit_transform(data)
        norm_data = min_max_scaler.fit_transform(norm_data)
        return norm_data
        
    def quantize(self, data, quantize_level):
        return np.round(data*(1<<quantize_level)) / (1<<quantize_level)
    
def location_cal(sensor_positions, batch_features):
    NumChannels = batch_features.shape[1]
    location_day = []

    b_max = batch_features.max(-1)
    b_min = batch_features.min(-1)
    amplitudes = b_max-b_min

    amplitudes =np.square(amplitudes)
    amplitudes = np.square(amplitudes)
    sum_square_amplitute=np.sum(amplitudes,axis=1)

    location_day=[]
    for ij in range(sensor_positions.shape[1]):
        x=np.dot(sensor_positions[:, ij] , amplitudes.T)
        x=np.divide(x, sum_square_amplitute)
        location_day.append(x)

    location_day=np.array(location_day).T
    return location_day

def location_cal_group(sensor_positions, batch_features,group_id):
    group_batch = sensor_positions[:,-1]
    location_day=np.zeros((batch_features.shape[0],3))
    for i in np.unique(group_batch):
        care_loc = np.where(group_batch==i)[0]
        look_spike_loc = np.nonzero(np.in1d(group_id, care_loc))[0]
        location_day_batch = location_cal(sensor_positions[care_loc,:], batch_features[look_spike_loc,:,:][:,care_loc,:])
        location_day[look_spike_loc,:] = location_day_batch
    return location_day