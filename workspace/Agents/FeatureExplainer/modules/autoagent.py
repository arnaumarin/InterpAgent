from pathlib import Path
import numpy as np

from spikeinterface.core.npzsortingextractor import NpzSortingExtractor
import spikeinterface.extractors as se
                

from .spike_detection import detect_spike_test
from .autosort.clssimp import clssimp
from .data_loader import Custom_Norm
from .autosort.assistant import (Assistant, stats)
import torch
import torch.nn as nn
from tqdm import tqdm
from spikeinterface.core.npzsortingextractor import NpzSortingExtractor

from .sma_fun import SMA
from .sma_fun import SMA_online

import os

class AutoAgent():
    def __init__(self, params, dirs):
        self.trained_net = Path(f"{dirs.trained_model_path/params.model_name}.pt")
        if not self.trained_net.exists(): raise KeyError('Trained Autosort not found')
        print(f"Autosort model successfully loaded from: {self.trained_net}")
        input_size = params.input_size
        output_size = params.output_size
        device=params.device
        self.model = clssimp(input_size, output_size).to(device)
        self.model.load_state_dict(torch.load(f"{self.trained_net}"))
        self.assistant = Assistant(net=self.model)
        self.stats = stats()

        self.shank_position = np.load(dirs.sp_path)
        self.num_neurons = output_size-1
        
        self.buffer = 200
        self.h_buffer = self.buffer//2
        self.n_block = 200
        
        self.unit_ids = np.arange(self.num_neurons)
        self.num_seg = 1

    def initialize(self):
        self.spike_time = []
        self.spike_label = []
        self.feature = []
        self.trajectory = []
        self.sma = SMA_online(self.num_neurons)
        #self.sma = SMA(self.num_neurons)
        
    def get_recording(self, recording, tp=1):
        freq = recording.get_sampling_frequency()
        self.freq = freq
        day_length = recording.get_total_samples()
        n_block = int(day_length//(tp*freq))
        residue = day_length%(tp*freq)
        n_iter = n_block+1 if residue!=0 else n_block

        yield None
        for i in tqdm(range(n_iter)):
            self.start = int((i*freq*tp)-self.buffer)
            self.end = int((i+1)*tp*freq)
            self.start_p = int((i*freq*tp)-self.h_buffer)
            self.end_p = int((i+1)*tp*freq-self.h_buffer)
            if i == 0:
                self.start = 0
                self.start_p = 0
            if i == n_block: 
                self.end = day_length
                self.end_p = day_length
            yield recording.frame_slice(self.start,self.end).get_traces()
        yield None

    def run_sorting(self, recording, sorting_path, thsd=4, tp=1):
        print("\nSorting start\n")
        rec = self.get_recording(recording, tp=tp)
        next(rec)
        while True: #
            raw_trace = next(rec) #
            if raw_trace is None: #
                break #
            processed_raw_trace, spike_time, spike_label, sma_data = self.process(raw_trace, thsd) #
            #yield raw_trace[self.start_p-self.start: self.end_p-self.end or None],spike_time,spike_label,sma_data
        sorting = self.save_sorting(sorting_path) #
        print("\nSorting finished.\n")
        return sorting
    
    def process(self, trace_seg, thsd):
        trace_length = trace_seg.shape[0]
        start = self.h_buffer
        end = trace_length - self.h_buffer
        
        spikes = detect_spike_test(trace_seg,thr_min = thsd, thr_max=30, 
                    distance=3, ch_max_simul_firing = 5,wlen=5, prominence=10)
        
        X_spiketrain_time = np.where(spikes)[0]

        mask = (X_spiketrain_time > start) & (X_spiketrain_time < end)
        X_spiketrain_time = X_spiketrain_time[mask]
        Y_spiketrain_id_final = np.where(spikes)[1][mask]
        
        assert X_spiketrain_time.shape[0] == Y_spiketrain_id_final.shape[0]

        n_spike = Y_spiketrain_id_final.shape[0]
        if n_spike==0:return
        
        for time_range in np.arange(-10,20):
            if time_range==-10:
                waveform = trace_seg[X_spiketrain_time+time_range,:]
            else:
                waveform = np.dstack((waveform, trace_seg[X_spiketrain_time+time_range,:] ))
                
        custom_norm = Custom_Norm(self.shank_position, Y_spiketrain_id_final)
        
        pred_loc = self.shank_position[:,:-1][Y_spiketrain_id_final]
        waveform_multi = custom_norm.group_shank(waveform)
        waveform_multi = waveform_multi.reshape(n_spike, -1)
        waveform_single = waveform[np.arange(n_spike), np.array(Y_spiketrain_id_final).astype('int'), :]
        
        test_x = np.concatenate((waveform_multi,waveform_single,pred_loc), axis=1)
        test_x = torch.tensor(test_x)
        
        output = self.assistant.real(test_x)
        
        pred_label = np.argmax(output,axis=1)
        
        noise_mask = pred_label == self.num_neurons
        spike_time = self.start+X_spiketrain_time[~noise_mask]
        spike_label = pred_label[~noise_mask]
        feauture = test_x[~noise_mask]
        assert feauture.shape[0] == spike_time.shape[0]
        self.feature.extend(feauture)
        self.spike_time.extend(spike_time)
        self.spike_label.extend(spike_label)

        sma_data = self.process_trajectory(spike_time, spike_label)  
        self.trajectory.extend(sma_data.T)

        raw_trace = trace_seg[self.start_p-self.start:self.end_p-self.start]
        return raw_trace, spike_time, spike_label, sma_data
        
    def process_trajectory(self, spike_time, spike_label):
        session_len = self.end_p - self.start_p
        spike_train = np.zeros((self.num_neurons, session_len))
        ax1 = np.array(spike_label).astype(int)
        ax2 = np.array(spike_time - self.start_p).astype(int)
        spike_train[ax1,ax2] = 1
        sma_data = self.sma.apply(spike_train, n_block=self.n_block)

        return sma_data
        
    def save_sorting(self, sorting_path):
        sorter_output_path = os.path.join(sorting_path, "sorter_output")
        os.makedirs(sorter_output_path,exist_ok=True)
        firing_save_path = os.path.join(sorter_output_path, "firings.npz")

        self.spike_time = np.array(self.spike_time)
        self.spike_label = np.array(self.spike_label)
        
        firings = {'unit_ids': self.unit_ids,
                'num_segment': np.array([self.num_seg]),
                'sampling_frequency': np.array([self.freq],dtype=np.float64),
                'spike_indexes_seg0': self.spike_time.astype(np.int64),
                'spike_labels_seg0': self.spike_label.astype(np.int64)}
        
        np.savez(firing_save_path, **firings)

        self.trajectory.extend(self.sma.get_edge().T)
        self.trajectory = np.array(self.trajectory).T
        self.feature = np.array(self.feature)
        np.save(os.path.join(sorter_output_path,'trajectory.npy'),self.trajectory)
        np.save(os.path.join(sorter_output_path,'feauture.npy'),self.feature)
        from spikeinterface.extractors import read_npz_sorting
        non_empty_unit_ids = self.unit_ids[np.isin(self.unit_ids,self.spike_label)]
        sorting = read_npz_sorting(firing_save_path)
        sorting = sorting.select_units(unit_ids=non_empty_unit_ids)
        return sorting


from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Range1d, FixedTicker, BoxAnnotation, Segment
from bokeh.layouts import gridplot
import numpy as np
from tornado import gen

class BokehRealtime:
    def __init__(self, recording, tp, time_bin, channel, num_neurons, n_block):
        self.tp = tp
        self.time_bin = time_bin
        self.freq = recording.get_sampling_frequency()
        self.total_samples = int(recording.get_total_samples())
        self.channel = channel
        self.num_neurons = num_neurons
        self.n_block = n_block

        self.rollover = time_bin * self.freq
        self.count = 0
        self.trace_len = 0
        self.n_trial = 0

        self._init_sources()
        self._init_figures()
        self.layout = gridplot([[self.p1], [self.p2], [self.p3]])

    def _init_sources(self):
        self.source1 = ColumnDataSource(data={'x': [], 'y1': [], 'y2': [], 'y3': [], 'y4': []})
        self.source2 = ColumnDataSource(data={'x0': [], 'x1': [], 'y0': [], 'y1': []})
        self.source3 = ColumnDataSource(data={'x': [], 'y': [], 'color': []})

    def _init_figures(self):
        self.p1 = figure(x_axis_label="Time(s)", y_axis_label="Channels", width=1500,
                         height=100 * len(self.channel), x_range=Range1d(0, self.rollover))

        trace_colors = ["#4B0082", "#6A5ACD", "#4169E1", "#4682B4"]  # dark purple to blue shades
        y_offsets = [150, 50, -50, -150]

        for i, (color, offset, ch) in enumerate(zip(trace_colors, y_offsets, self.channel)):
            self.p1.line('x', f'y{i+1}', source=self.source1, line_width=2, color=color, legend_label=f"Ch {ch}")

        self.p1.y_range.start = -250
        self.p1.y_range.end = 250
        self.p1.yaxis.ticker = y_offsets
        self.p1.yaxis.major_label_overrides = {offset: f"Ch {ch}" for offset, ch in zip(y_offsets, self.channel)}

        self.p2 = figure(x_axis_label="Time(s)", y_axis_label="Neuron labels", width=1500,
                         height=15 * self.num_neurons, title='Spike Train', x_range=Range1d(0, self.rollover))
        self.p2.segment('x0', 'y0', 'x1', 'y1', source=self.source2,
                        line_color="black", line_alpha=0.8, line_width=1)
        self.p2.y_range.start = 0
        self.p2.y_range.end = self.num_neurons

        self.p3 = figure(x_axis_label="Time(s)", y_axis_label="SMA value", width=1500, height=250,
                         title='Trajectory', x_range=Range1d(0, self.rollover))
        self.p3.circle('x', 'y', source=self.source3, color='color', size=4)
        self.p3.y_range.start = -4
        self.p3.y_range.end = 1

    def set_trigger(self, trigger_time, ms_before=1, ms_after=1):
        trig_on = np.array(trigger_time)*self.freq
        trig_on = trig_on.astype(int)
        self.custom_trigger = np.zeros(self.total_samples, dtype=bool)
        offsets = np.arange(int(-ms_before*self.freq), int(ms_after*self.freq))
        expanded = trig_on[:, None] + offsets
        all_indices = np.clip(expanded.ravel(), 0, self.total_samples - 1)
        self.custom_trigger[all_indices] = True

    def create_update_func(self):
        @gen.coroutine
        def update(raw_trace=None, spike_time=None, spike_label=None, trajectory=None):
            new_trace_len = raw_trace.shape[0]
            start = self.trace_len
            end = self.trace_len + new_trace_len
            all_x = np.arange(start, end)
            raw_trace = raw_trace[:, self.channel]

            if self.count % self.time_bin == 0:
                self._reset_sources()
                self._update_ranges()

            self.count += self.tp
            self.trace_len += new_trace_len

            self.source1.stream({
                'x': all_x,
                'y1': raw_trace[:, 0] + 150,
                'y2': raw_trace[:, 1] + 50,
                'y3': raw_trace[:, 2] - 50,
                'y4': raw_trace[:, 3] - 150
            })

            if spike_time is not None and spike_label is not None:
                self.source2.stream({
                    'x0': spike_time,
                    'x1': spike_time,
                    'y0': spike_label - 0.4,
                    'y1': spike_label + 0.4
                })

            if trajectory is not None:
                n_trial = trajectory.shape[1]
                trial_x = np.arange(self.n_trial+1, self.n_trial+1 + n_trial) * self.n_block
                trial_y = trajectory[0, :]
                if hasattr(self,"custom_trigger"):
                    colors = ["#FF4500" if self.custom_trigger[int(x)] else "#1E90FF" for x in trial_x]
                else:
                    colors = ["#1E90FF" for _ in trial_x]
                self.source3.stream({'x': trial_x, 'y': trial_y, 'color': colors})
                self.n_trial += n_trial

        return update

    def _reset_sources(self):
        self.source1.data = {key: [] for key in ['x', 'y1', 'y2', 'y3', 'y4']}
        self.source2.data = {key: [] for key in ['x0', 'x1', 'y0', 'y1']}
        self.source3.data = {key: [] for key in ['x', 'y', 'color']}

    def _update_ranges(self):
        for p in [self.p1, self.p2, self.p3]:
            p.x_range.start = self.count * self.freq
            p.x_range.end = self.count * self.freq + self.rollover
            ticks = np.linspace(p.x_range.start, p.x_range.end, self.time_bin + 1)
            p.xaxis.ticker = FixedTicker(ticks=ticks)
            p.xaxis.major_label_overrides = {t: f'{t/self.freq:0.1f}' for t in ticks}

    @staticmethod
    def get_server():
        port=5007
        bokeh_url = f"http://localhost:{port}/"
        def run_server(autosagent, bokeh_realtime, thsd, rec, sorting_path, port=5007):
            from bokeh.server.server import Server
            from tornado.ioloop import IOLoop
            update = bokeh_realtime.create_update_func()

            def modify_doc(doc):
                def periodic_update():
                    raw_trace = next(rec)
                    if raw_trace is None:
                        doc.add_next_tick_callback(lambda: doc.remove_periodic_callback(callback_handle))
                        server.io_loop.stop()
                        return
                    processed_raw_trace, spike_time, spike_label, sma_data = autosagent.process(raw_trace, thsd=thsd)
                    update(processed_raw_trace, spike_time, spike_label, sma_data)

                callback_handle = doc.add_periodic_callback(periodic_update, 500)
                doc.add_root(bokeh_realtime.layout)
            
            server = Server({'/': modify_doc}, port=port, allow_websocket_origin=["*"])
            server.start()
            #server.io_loop.add_callback(server.show, '/')
            #server.io_loop.start()
            IOLoop.current().start()

            # server = Server({'/': modify_doc}, port=port)
            # server.io_loop.add_callback(server.show, '/')
            # server.io_loop.start()

            sorting = autosagent.save_sorting(sorting_path)
            return sorting

        return run_server, bokeh_url