import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import time
import json
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
    # set 10 cores for training
    torch.set_num_threads(10)


    print(colored('Using device:', 'green'), device)

    SEED = 12345
    _=np.random.seed(SEED)
    _=torch.manual_seed(SEED)

    epoch_num   = 1
    batch_size  = 256
    enable_save = False
    # training_event_size = 110409
    training_event_size = 1000
    test_event_size = 400

    train_datapath = "./Node_Edge_Classification/data/if-graph-train.h5"
    test_datapath = "./Node_Edge_Classification/data/if-graph-test.h5"

    voxel_data_train = ShowerVoxels(file_path = train_datapath)
    voxel_data_test  = ShowerVoxels(file_path = test_datapath)
    example_voxel_data = voxel_data_train[0]
    print("Voxel data keys: ", example_voxel_data.keys())
    print("Voxel data size: ", len(voxel_data_train))


    # * --- Generate global features from voxel data --- *
    print(colored('Generating training global features from voxel data ...', 'green'))
    train_u_features = Net_MetaLayer.extractGlobalFeatures(voxel_data_train, training_event_size, batch_size)
    print(colored('Generating test global features from voxel data ...', 'green'))
    test_u_features = Net_MetaLayer.extractGlobalFeatures(voxel_data_test, test_event_size, batch_size)

    # * ----------------------------------------------- *

    print(colored('Loading training data from: ', 'green'), train_datapath)
    train_data = ShowerFeatures(file_path = train_datapath)
    # use only a slice of the training data
    train_data_slice = []
    print(colored('Slicing training data ...', 'green'))
    if training_event_size < 110409:
        for i in range(0, training_event_size):
            train_data_slice.append(train_data[i])
        print('Size of training data: ', len(train_data))


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

    if training_event_size < 110409:
        train_loader = GraphDataLoader(
            train_data_slice,
            num_workers = 0,
            shuffle     = False,
            batch_size  = batch_size
        )
    else:
        train_loader = GraphDataLoader(
            train_data,
            num_workers = 0,
            shuffle     = False,
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
    # try_pred = model(
    #     x = train_data[0].x, 
    #     edge_index = train_data[0].edge_index, 
    #     edge_attr  = train_data[0].edge_attr, 
    #     batch = batch_placeholder,
    #     u = u_placeholder)
    # print(try_pred)


    print(colored('Defining loss and optimizer ...', 'green'))
    loss_func = torch.nn.BCELoss()
    # loss_func =  Net_MetaLayer.DrielsmaLoss()
    loss_func.to(device)
    loss_edge_ratio = 0.5
    loss_node_ratio = 1 - loss_edge_ratio
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0025)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()

    loss_array_node = []
    loss_array_edge = []
    test_acc_node = []
    test_acc_edge = []


    print(colored('Training ...', 'cyan', attrs=['bold']))
    # ! training loop
    for epoch in range(epoch_num):
        epoch_start_time = time.time()
        batch_cnt = 0
        for batch in train_loader:
            batch_cnt += 1
            start_time = time.time()
            optimizer.zero_grad()
            pred_y, pred_edge_label = model(
                x = batch.x,
                edge_index = batch.edge_index,
                edge_attr  = batch.edge_attr,
                batch = batch.batch,
                u = train_u_features[batch_cnt - 1]
            )
            node_labels = batch.y.reshape(len(batch.y), 1).float()
            node_labels = node_labels.to(device)
            loss_node   = loss_func(pred_y, node_labels)
            # loss_node = Net_MetaLayer.DrielsmaLoss(pred_y, node_labels)

            edge_labels = batch.edge_label.reshape(len(batch.edge_label), 1).float()
            edge_labels = edge_labels.to(device)
            loss_edge   = loss_func(pred_edge_label, edge_labels)
            # loss_edge = Net_MetaLayer.DrielsmaLoss(pred_edge_label, edge_labels)
            loss = loss_node + loss_edge
            # loss = loss_node_ratio * loss_node + loss_edge_ratio * loss_edge
            loss.backward()
            optimizer.step()
            loss_array_node.append(loss_node.item())
            loss_array_edge.append(loss_edge.item())
            end_time = time.time()
            loss_2p3_str = str(loss_edge.item())[:5]
            batch_time = end_time - start_time
            batch_time = np.round(batch_time*1000, 1)
            time_2p2_str = str(batch_time)
            print('[ Epoch: ', epoch, ' Edge Loss: ', loss_2p3_str, ' Time[ms]: ', time_2p2_str, 'batch:' , batch_cnt, '/', train_batch_num, ' ]',end='\r')

        local_acc_node = []
        local_acc_edge = []
        local_last_pred_y_true = []
        local_last_pred_y_false = []
        local_last_pred_edge_true = []
        local_last_pred_edge_false = []
        local_last_pred_edge_label = []
        # ! Start testing
        batch_test_cnt = 0
        for batch in test_loader:
            batch_test_cnt += 1
            pred_y, pred_edge_label = model(
                x = batch.x,
                edge_index = batch.edge_index,
                edge_attr  = batch.edge_attr,
                batch = batch.batch,
                u = test_u_features[batch_test_cnt - 1]
            )
            
            pred_y = pred_y.detach().to('cpu').numpy()
            # pred_y = pred_y[:, 1]
            real_y = batch.y.reshape(len(batch.y), 1).numpy()
            for i in range(len(pred_y)):
                if real_y[i].item() == 1:
                    local_last_pred_y_true.append(pred_y[i].item())
                else:
                    local_last_pred_y_false.append(pred_y[i].item())
            pred_y = np.round(pred_y)

            pred_edge_label = pred_edge_label.detach().to('cpu').numpy()
            # pred_edge_label = pred_edge_label[:, 1]
            real_edge_label = batch.edge_label.reshape(len(batch.edge_label), 1).numpy()
            for i in range(len(pred_edge_label)):
                if real_edge_label[i].item() == 1:
                    local_last_pred_edge_true.append(pred_edge_label[i].item())
                else:
                    local_last_pred_edge_false.append(pred_edge_label[i].item())
            pred_edge_label = np.round(pred_edge_label)
            # accuracy is the number of correct predictions divided by the number of predictions
            acc_y = np.sum(pred_y == real_y) / len(batch.y)
            acc_edge = np.sum(pred_edge_label == batch.edge_label.reshape(len(batch.edge_label), 1).numpy()) / len(batch.edge_label)

            local_acc_node.append(acc_y)
            local_acc_edge.append(acc_edge)
        test_acc_edge.append(np.mean(local_acc_edge))
        test_acc_node.append(np.mean(local_acc_node))
        epoch_time = (time.time() - epoch_start_time)
        epoch_time = np.round(epoch_time, 2)
        print('\n-        ', epoch, ' Node accuracy: ', str(np.mean(local_acc_node))[:5], ' Edge accuracy: ', str(np.mean(local_acc_edge))[:5], ' Time[sec]: ', str(epoch_time))

    local_time = time.localtime(time.time())
    # save model
    model_name = './Node_Edge_Classification/temp/model' + str(local_time.tm_year) + str(local_time.tm_mon) + str(local_time.tm_mday) + str(local_time.tm_hour) + str(local_time.tm_min) + str(local_time.tm_sec) + '.pt'
    if enable_save:
        torch.save(model.state_dict(), model_name)
    # save loss into json
    loss_dict = {
        'loss_node': loss_array_node,
        'loss_edge': loss_array_edge,
        'test_acc_node': test_acc_node,
        'test_acc_edge': test_acc_edge
    }
    json_name = './Node_Edge_Classification/temp/loss' + str(local_time.tm_year) + str(local_time.tm_mon) + str(local_time.tm_mday) + str(local_time.tm_hour) + str(local_time.tm_min) + str(local_time.tm_sec) + '.json'
    if enable_save:
        with open(json_name, 'w') as f:
            json.dump(loss_dict, f)
    
    # * Create loss figure
    if enable_save:
        figure_train = plt.figure(figsize=(10, 5), dpi=1000)
    else:
        figure_train = plt.figure(figsize=(10, 5))
    plt.plot(loss_array_node, label='Node loss', color='#04B5BBEE')
    plt.plot(loss_array_edge, label='Edge loss', color='#DF0345EE')
    figure_info = 'Training with ' + str(epoch_num) + ' epochs, ' + str(training_event_size) + ' events per epoch, ' + str(batch_size) + ' events per batch'
    plt.title(figure_info)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.legend(loc='lower left')
    # sub y axis
    ax1 = plt.gca()
    ax2 = plt.twinx()
    # unify x axis
    x_acc = np.arange(0, len(test_acc_node), 1)
    for val in x_acc:
        x_acc[val] = val * train_batch_num + train_batch_num/2
    ax2.plot(x_acc, test_acc_node, label='Node accuracy', color='#04B5BBEE', marker='o', linestyle='--')
    ax2.plot(x_acc, test_acc_edge, label='Edge accuracy', color='#DF0345EE', marker='o', linestyle='--')
    ax2.set_ylabel('Accuracy')
    ax2.legend(loc='upper right')
    # set x axis range
    plt.xlim(0, len(loss_array_node))

    ax1.set_ylim(0, 0.8)
    ax2.set_ylim(0.8, 1)
    plt.grid()
    
    if enable_save:
        plt.savefig('./Node_Edge_Classification/pics/res' + str(local_time.tm_year) + str(local_time.tm_mon) + str(local_time.tm_mday) + str(local_time.tm_hour) + str(local_time.tm_min) + str(local_time.tm_sec) + '.png')
    else:
        plt.show()

    # * Create node hist figure
    if enable_save:
        figure_node_hist = plt.figure(figsize=(8, 8), dpi=600)
    else:
        figure_node_hist = plt.figure(figsize=(8, 8))

    print('False node value number: ', len(local_last_pred_y_false))
    print('True node value number: ', len(local_last_pred_y_true))
    print('False edge value number: ', len(local_last_pred_edge_false))
    print('True edge value number: ', len(local_last_pred_edge_true))

    print('FNode range: ', np.min(local_last_pred_y_false), np.max(local_last_pred_y_false))
    print('TNode range: ', np.min(local_last_pred_y_true), np.max(local_last_pred_y_true))
    print('FEdge range: ', np.min(local_last_pred_edge_false), np.max(local_last_pred_edge_false))
    print('TEdge range: ', np.min(local_last_pred_edge_true), np.max(local_last_pred_edge_true))
    # normalized histogram
    plt.hist(local_last_pred_y_true,  bins=50, label='Node True',  range=(0,1), ec='#04B5BBEE', lw=2, histtype='step')
    plt.hist(local_last_pred_y_false, bins=50, label='Node False', range=(0,1), ec='#DF0345EE', lw=2, histtype='step')
    plt.hist(local_last_pred_edge_true, bins=50, label='Edge True', range=(0,1), ec='#1C43C3EE', lw=2, histtype='step')
    plt.hist(local_last_pred_edge_false, bins=50, label='Edge False', range=(0,1), ec='#BB8104EE', lw=2, histtype='step')

    # log y axis
    plt.yscale('log')
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.xlim(0, 1)
    plt.legend(loc='upper center')
    plt.grid()
    if enable_save:
        plt.savefig('./Node_Edge_Classification/pics/node_hist' + str(local_time.tm_year) + str(local_time.tm_mon) + str(local_time.tm_mday) + str(local_time.tm_hour) + str(local_time.tm_min) + str(local_time.tm_sec) + '.png')
    else:
        plt.show()

    


