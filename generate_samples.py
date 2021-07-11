#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


def generate_train_val_test(
    input_dir, input_filename, output_dir, NSess, NSub, input_horizon=30, output_horizon=30, 
    scaling='n', save_data=True, train_prop=0.8, test_prop=0.1, NROIs=None,
    ):
    
    '''
    Generate train, validation and test data samples from fMRI timecourses.
    
    input_dir: directory with structure input_dir/Session_{nSess}/
               with nSess running from 1 to NSess
    input_filename: .txt file in: input_dir/Session_{nSess}
                    with name: input_filename{nSub}
                    with nSub running from 1 to NSub
                    expected dimension: ROIs x Samples
    output_dir: directory where train.npz, val.npz, test.npz are saved to
                containing model inputs x and tragets y
                
    x: (num_samples, input_length, num_nodes, feature_dim)
    y: (num_samples, output_length, num_nodes, feature_dim)
    '''
    
    Sub_list = list(range(1, NSub+1))  # list with subjects from 1 to NSub        
    Sess_list = list(range(1, NSess+1))

    # Define offsets.
    # 0 is the latest observed sample.
    x_offsets = np.sort(np.concatenate((np.arange(-(input_horizon-1), 1, 1),)))
    # Predict the next timesteps.
    y_offsets = np.sort(np.arange(1, (output_horizon+1), 1))

    x_train, y_train, x_val, y_val, x_test, y_test = [], [], [], [], [], [] 
    for nSess in Sess_list:  # Iterate through sessions
        sessiondir = 'session_' + str(nSess) + '/'
        for nSub in Sub_list:  # Iterate through subjects
            subfile = input_filename + str(nSub) +'.txt'
            filename =  input_dir + sessiondir + subfile
            print("Load: " + filename)
            timeseries = np.loadtxt(filename, delimiter=",", dtype='float32')
            if scaling == 'n':  # scale values between 0 and 1              
                timeseries = (timeseries - timeseries.min().min())/(timeseries.max().max() - timeseries.min().min())
            elif scaling == 'z':  # standardize values
                timeseries = timeseries - timeseries.mean()
                timeseries = timeseries / timeseries.std()
            else: 
                pass
            timeseries = timeseries.T  # Now has shape samples x ROIs
            num_samples, num_nodes = timeseries.shape  
            
            if NROIs:  # Select only first nodes                
                timeseries = timeseries[:,:NROIs]

            # Load data, t is the index of the last observation.
            min_t = abs(min(x_offsets))
            max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
            x, y = [], []  # initialize as list again
            for t in range(min_t, max_t):
                x_t = timeseries[t + x_offsets, ...]
                y_t = timeseries[t + y_offsets, ...]
                x.append(x_t)
                y.append(y_t)

            x = np.stack(x, axis=0)
            y = np.stack(y, axis=0)  

            # Add feature dimension.
            x = np.expand_dims(x, axis=-1)
            y = np.expand_dims(y, axis=-1)
            
            # Compute number of train, val, test samples.
            num_snippets = x.shape[0]
            num_test = round(num_snippets * test_prop)
            num_train = round(num_snippets * train_prop)
            num_val = num_snippets - num_test - num_train    
            
            # Split data.
            # Train.
            x_train.append(x[:num_train]), y_train.append(y[:num_train])
            # Val.
            x_val.append(x[num_train: num_train + num_val]), y_val.append(y[num_train: num_train + num_val])       
            # Test.
            x_test.append(x[-num_test:]), y_test.append(y[-num_test:])

    x_train = np.concatenate(x_train, axis=0)  # Concatenate all samples along first dimension.    
    y_train = np.concatenate(y_train, axis=0)
    x_val = np.concatenate(x_val, axis=0)
    y_val = np.concatenate(y_val, axis=0)
    x_test = np.concatenate(x_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    print('### SAMPLES ###')
    print('Using {} sessions from {} subjects.'.format(len(Sess_list), len(Sub_list)))    
    print('Per session: {:5} training samples, {:5} validation samples, {:5} testing samples.'.format(num_train, num_val, num_test))  
    print('In total:    {:5} training samples, {:5} validation samples, {:5} testing samples.'.format(x_train.shape[0], x_val.shape[0], x_test.shape[0]))
    
    # Save results.
    if save_data:
        print('### SAVE DATA ###')
        print('Save in: ' + output_dir)
        for cat in ["train", "val", "test"]:
            _x, _y = locals()["x_" + cat], locals()["y_" + cat]
            print(cat, "x: ", _x.shape, "y:", _y.shape)
            np.savez_compressed(
            os.path.join(output_dir, "%s.npz" % cat),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
            )
    print('Done.')
                
    return x_train, y_train, x_val, y_val, x_test, y_test


def main(args):
    print("Generating training data.")
    generate_train_val_test(input_dir=args.input_dir, input_filename=args.input_filename, output_dir=args.output_dir, 
                            NSess=args.NSess, NSub=args.NSub, input_horizon=args.input_horizon, output_horizon=args.output_horizon)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", type=str, default="./MRI_data/fMRI_sessions/", help="Directory where fMRI timecourses are stored in, with subdirectories session_1, session_2, etc."
    )
    parser.add_argument(
        "--input_filename", type=str, default="artificial_timeseries_sub_", help="Names of the fMRI timecourse data."
    )
    parser.add_argument(
        "--output_dir", type=str, default="./MRI_data/training_samples/", help="Output directory."
    )
    parser.add_argument(
        "--NSess", type=int, default="2", help="Specify number of fMRI sessions."
    )    
    parser.add_argument(
        "--NSub", type=int, default="10", help="Specify number of subjects."
    )    
    parser.add_argument(
        "--input_horizon", type=int, default="30", help="Number of timesteps for model input."
    )     
    parser.add_argument(
        "--output_horizon", type=int, default="30", help="Number of timesteps for model forecasting horizon."
    )     
    args = parser.parse_args()
    main(args)