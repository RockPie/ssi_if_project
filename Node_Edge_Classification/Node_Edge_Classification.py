import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import time
import Net_Tri_EdgeCov
import Net_MetaLayer
from termcolor import colored
from torch.utils.data import DataLoader
from iftool.gnn_challenge import plot_scatter3d
from iftool.gnn_challenge import ShowerVoxels
from iftool.gnn_challenge import ShowerFeatures
from torch_geometric.loader import DataLoader as GraphDataLoader
from torch_geometric.nn import MessagePassing, GCNConv, global_mean_pool, EdgeConv
from torch.nn import Linear, Parameter, functional as F
from torch_geometric.utils import add_self_loops, degree

if __name__ == '__main__':
    print(colored('Script started', 'green'))
    torch.multiprocessing.set_start_method('spawn', force=True)

    # device = torch.device('mps')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(colored('Using device:', 'green'), device)

    SEED = 12345
    _=np.random.seed(SEED)
    _=torch.manual_seed(SEED)

    batch_size = 256
    training_event_size = 110409
    test_event_size = 5000

    train_datapath = "./Node_Edge_Classification/data/if-graph-train.h5"

    voxel_data = ShowerVoxels(file_path = train_datapath)
    example_voxel_data = voxel_data[0]
    # print('List of keys in a data element',example_voxel_data.keys())

    print(colored('Loading training data from: ', 'green'), train_datapath)
    train_data = ShowerFeatures(file_path = train_datapath)
    # use only a slice of the training data
    train_data_slice = []
    print(colored('Slicing training data ...', 'green'))
    for i in range(0, training_event_size):
        train_data_slice.append(train_data[i])
    print('Size of training data: ', len(train_data))

    test_datapath = "./Node_Edge_Classification/data/if-graph-test.h5"
    print(colored('Loading test data from: ', 'green'), test_datapath)
    test_data = ShowerFeatures(file_path = test_datapath)
    test_data_slice = []
    print(colored('Slicing test data ...', 'green'))
    for i in range(0, test_event_size):
        test_data_slice.append(test_data[i])
    print('Size of test data: ', len(test_data))

    example_train_data = train_data[0]
    print('Example training data: ', example_train_data)

    example_test_data = test_data[0]
    print('Example test data: ', example_test_data)

    # split train data into train and validation

    # ! create data loaders

    print(colored('Creating data loaders ...', 'cyan'))

    train_loader = GraphDataLoader(
        train_data_slice,
        num_workers = 0,
        shuffle     = True,
        batch_size  = batch_size
    )

    test_loader = GraphDataLoader(
        test_data_slice,
        num_workers = 0,
        shuffle     = False,
        batch_size  = batch_size
    )

    train_batch_num = len(train_loader)
    test_batch_num = len(test_loader)
    print('Number of batches in train loader: ', train_batch_num)
    print('Number of batches in test loader: ', test_batch_num)

    # ! our code here
    print(colored('Creating model ...', 'cyan'))
    model = Net_MetaLayer.SJN_Meta(train_data, device=device)
    model.to(device)

    batch_placeholder = None
    u_placeholder     = None

    print(colored('Testing model ...', 'green'))
    try_pred = model(
        x = train_data[0].x, 
        edge_index = train_data[0].edge_index, 
        edge_attr  = train_data[0].edge_attr, 
        batch = batch_placeholder,
        u = u_placeholder)
    # print(try_pred)


    print(colored('Defining loss and optimizer ...', 'green'))
    loss_func = torch.nn.BCELoss()
    loss_func.to(device)
    loss_edge_ratio = 0.5
    loss_node_ratio = 1 - loss_edge_ratio
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()

    loss_array_node = []
    loss_array_edge = []
    test_acc_node = []
    test_acc_edge = []
    epoch_num = 20

    print(colored('Training ...', 'cyan', attrs=['bold']))
    for epoch in range(epoch_num):
        batch_cnt = 0
        for batch in train_loader:
            batch_cnt += 1
            start_time = time.time()
            optimizer.zero_grad()
            pred_y, pred_edge_label = model(
                x = batch.x,
                edge_index = batch.edge_index,
                edge_attr  = batch.edge_attr,
                batch = batch_placeholder,
                u = u_placeholder
            )

            node_labels = batch.y.reshape(len(batch.y), 1).float()
            node_labels = node_labels.to(device)
            loss_node   = loss_func(pred_y, node_labels)

            edge_labels = batch.edge_label.reshape(len(batch.edge_label), 1).float()
            edge_labels = edge_labels.to(device)
            loss_edge   = loss_func(pred_edge_label, edge_labels)
            loss = loss_node_ratio * loss_node + loss_edge_ratio * loss_edge
            loss.backward()
            optimizer.step()
            loss_array_node.append(loss_node.item())
            loss_array_edge.append(loss_edge.item())
            end_time = time.time()
            loss_2p3_str = str(loss_edge.item())[:5]
            time_2p2_str = str((end_time - start_time)*1000)[:4]
            print('[ Epoch: ', epoch, ' Edge Loss: ', loss_2p3_str, ' Time[ms]: ', time_2p2_str, 'batch:' , batch_cnt, '/', train_batch_num, ' ]',end='\r')

        local_acc_node = []
        local_acc_edge = []
        for batch in test_loader:
            pred_y, pred_edge_label = model(
                x = batch.x,
                edge_index = batch.edge_index,
                edge_attr  = batch.edge_attr,
                batch = batch_placeholder,
                u = u_placeholder
            )
            pred_y = pred_y.detach().to('cpu').numpy()
            pred_y = np.round(pred_y)

            pred_edge_label = pred_edge_label.detach().to('cpu').numpy()
            pred_edge_label = np.round(pred_edge_label)
            # accuracy is the number of correct predictions divided by the number of predictions
            acc_y = np.sum(pred_y == batch.y.reshape(len(batch.y), 1).numpy()) / len(batch.y)
            acc_edge = np.sum(pred_edge_label == batch.edge_label.reshape(len(batch.edge_label), 1).numpy()) / len(batch.edge_label)
            local_acc_node.append(acc_y)
            local_acc_edge.append(acc_edge)
        test_acc_edge.append(np.mean(local_acc_edge))
        test_acc_node.append(np.mean(local_acc_node))
        print('\n-        ', epoch, ' Node accuracy: ', str(np.mean(local_acc_node))[:5], ' Edge accuracy: ', str(np.mean(local_acc_edge))[:5])

    print('Overallest node accuracy: ', np.mean(test_acc_node))
    print('Overallest edge accuracy: ', np.mean(test_acc_edge))
            
    figure_train = plt.figure(figsize=(10, 5), dpi=1000)
    plt.plot(loss_array_node, label='Node loss', color='#04B5BBEE')
    plt.plot(loss_array_edge, label='Edge loss', color='#DF0345EE')
    figure_info = 'Training with ' + str(epoch_num) + ' epochs, ' + str(training_event_size) + ' events per epoch, ' + str(batch_size) + ' events per batch'
    plt.title(figure_info)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.legend(loc='lower left')
    # sub y axis
    ax2 = plt.twinx()
    # unify x axis
    x_acc = np.arange(0, len(test_acc_node), 1)
    for val in x_acc:
        x_acc[val] = val * train_batch_num + train_batch_num/2
    ax2.plot(x_acc, test_acc_node, label='Node accuracy', color='#04B5BBEE', marker='o', linestyle='--')
    ax2.plot(x_acc, test_acc_edge, label='Edge accuracy', color='#DF0345EE', marker='o', linestyle='--')
    ax2.set_ylabel('Accuracy')
    ax2.legend(loc='center right')
    # set x axis range
    plt.xlim(0, len(loss_array_node))

    plt.ylim(0, max(loss_array_node) * 1.1)
    ax2.set_ylim(0.8, 1)
    plt.grid()
    local_time = time.localtime(time.time())
    plt.savefig('./Node_Edge_Classification/pics/res' + str(local_time.tm_year) + str(local_time.tm_mon) + str(local_time.tm_mday) + str(local_time.tm_hour) + str(local_time.tm_min) + str(local_time.tm_sec) + '.png')

