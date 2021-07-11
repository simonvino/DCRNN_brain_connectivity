import numpy as np
import matplotlib.pyplot as plt
import math


def plot_predictions(log_dir, dataset_dir, NROI=[1, 15], NSample=0, save_figure=True):
    '''
    Visualize model predictions. 
    
    log_dir: directory where model outputs were saved to
    dataset_dir: directory where data samples were saved to
    NROI: plot predictions from NROI[0] to NROI[1]
    NSample: number of test sample    
    '''

    test_data = np.load(dataset_dir + '/test.npz')
    outputs = np.load(log_dir + 'outputs.npz')
    outputs_predicitons = outputs['predictions'].transpose((1,0,2))  # To have shape (samples, time, ROI)
    outputs_groundtruth = outputs['groundtruth'].transpose((1,0,2))
    
    input_len=test_data['x'].shape[1]
    horizon=outputs_predicitons.shape[1]
       
    figurename=('DCRNN_predictions_test_sample_{}'.format(NSample)) 
    figurename= figurename + '_ROI{}-{}'.format(NROI[0],NROI[1])
    Nsubfigs = NROI[1]-NROI[0]+1
    fig=plt.figure(num=figurename, figsize=(27, Nsubfigs)) 
    fig.subplots_adjust(top=0.93, right = 0.88, left = 0.12, wspace = 0.15,
                    hspace=0.4, bottom = 0.07 )
    
    t_in=np.linspace(1,input_len,input_len)
    t_out=np.linspace(input_len+1,input_len+horizon,horizon)
    
    Nrows=math.ceil((Nsubfigs)/3)
        
    # Creat plots.
    for nROI in range(NROI[0]-1, NROI[1]):
        
        ax=fig.add_subplot(Nrows,3,nROI+1)

        plt.plot(t_in, test_data['x'][NSample,:,nROI,0], linestyle='-', marker='o', linewidth=4,
                 markersize='2.3', markeredgecolor='black', color=(0, 0.9, 0.4, 0.5))
        plt.plot(t_out, outputs_groundtruth[NSample,:,nROI], linestyle='-', marker='o', linewidth=4,
                 markersize='2.3', markeredgecolor='black', color=(0, 0.9, 0.4, 0.5), label='Truth')
        plt.plot(t_out, outputs_predicitons[NSample,:,nROI], linestyle='-', marker='o',
                 markersize='2.3', markeredgecolor='black', color='darkblue', label='Prediction')
        plt.axvline(x=(input_len+0.5), color='black', linewidth='0.5')

        plt.legend()
        plt.title('ROI #{}'.format(nROI+1))

        ax.set_xlabel('TRs')
        ax.set_ylabel('BOLD Signal')
    
    plt.suptitle('Predictions (Sample #{})'.format(NSample)) 

    if save_figure:
        print('Save figure in: ', log_dir)  
        plt.savefig(log_dir + figurename + '.png', dpi=100)