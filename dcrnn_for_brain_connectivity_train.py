from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import numpy as np
import yaml

from lib.utils import load_graph_data
from lib.plot_functions import plot_predictions
from model.dcrnn_supervisor import DCRNNSupervisor

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Select GPU.


def main(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f)

        SC_mx = load_graph_data(supervisor_config)  # Load structural connectivity matrix.

        tf_config = tf.ConfigProto()
        if args.use_cpu_only:
            tf_config = tf.ConfigProto(device_count={'GPU': 0})
        tf_config.gpu_options.allow_growth = True
        with tf.Session(config=tf_config) as sess:
            supervisor = DCRNNSupervisor(adj_mx=SC_mx, **supervisor_config)
            supervisor.train(sess=sess)
            
            if args.save_predictions:
                outputs, _ = supervisor.evaluate(sess=sess)   
                
                print('Save outputs in: ', supervisor._log_dir)
                np.savez(supervisor._log_dir + '/outputs', 
                         predictions=outputs['predictions'], 
                         groundtruth=outputs['groundtruth'])  
                
                plot_predictions(log_dir=supervisor._log_dir, 
                                 dataset_dir=supervisor_config['data']['dataset_dir'])                



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default=None, type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--save_predictions', default=True, type=bool, 
                        help='Save predictions of the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, 
                        help='Set to true to only use cpu.')
    args = parser.parse_args()
    main(args)
