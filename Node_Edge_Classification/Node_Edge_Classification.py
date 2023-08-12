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

print(colored('Script started', 'green'))
torch.multiprocessing.set_start_method('spawn')

# device = torch.device('mps')
# torch.set_num_threads(20)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


SEED = 12345
_=np.random.seed(SEED)
_=torch.manual_seed(SEED)

train_datapath = "./Node_Edge_Classification/data/if-graph-train.h5"

voxel_data = ShowerVoxels(file_path = train_datapath)
example_voxel_data = voxel_data[0]

print('List of keys in a data element',example_voxel_data.keys(),'\n')

print(colored('Loading training data from: ', 'green'), train_datapath)
train_data = ShowerFeatures(file_path = train_datapath)
# use only a slice of the training data
train_data_slice = []
print(colored('Slicing training data ...', 'green'))
for i in range(0, 10000):
    train_data_slice.append(train_data[i])
print('Size of training data: ', len(train_data))

test_datapath = "./Node_Edge_Classification/data/if-graph-test.h5"
print(colored('Loading test data from: ', 'green'), test_datapath)
test_data = ShowerFeatures(file_path = test_datapath)
test_data_slice = []
print(colored('Slicing test data ...', 'green'))
for i in range(0, 4000):
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
    shuffle     = True,
    batch_size  = 128
)

test_loader = GraphDataLoader(
    test_data_slice,
    shuffle     = False,
    batch_size  = 128
)

train_batch_num = len(train_loader)
test_batch_num = len(test_loader)
print('Number of batches in train loader: ', train_batch_num)
print('Number of batches in test loader: ', test_batch_num)

print(colored('Testing data loaders ...', 'green'))
tstart=time.time()
num_iter=1280
ctr=num_iter
for batch in train_loader:
    ctr -= 128
    if ctr <= 0: 
        break
print((time.time()-tstart)/num_iter,'[s/iteration]')

# ! our code here
print(colored('Creating model ...', 'cyan'))
model = Net_MetaLayer.SJN_Meta(train_data, device=device)
# model = Net_Tri_EdgeCov.SJN_NTE(train_data, device=device)

model.to(device)

print(colored('Testing model ...', 'green'))
try_pred = model(
    x = train_data[0].x, 
    edge_index = train_data[0].edge_index, 
    edge_attr  = train_data[0].edge_attr, 
    batch = train_data[0].batch,
    u = None)
# print(try_pred)


print(colored('Defining loss and optimizer ...', 'green'))
loss_func = torch.nn.BCELoss()
loss_edge_ratio = 0.5
loss_node_ratio = 1 - loss_edge_ratio
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.train()

loss_array = []
test_acc_y = []
test_acc_edge = []
epoch_num = 1

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
            batch = batch.batch,
            u = None
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
        loss_array.append(loss_edge.item())
        end_time = time.time()
        loss_2p3_str = str(loss_edge.item())[:5]
        time_2p2_str = str((end_time - start_time)*1000)[:4]
        print('[ Epoch: ', epoch, ' Edge Loss: ', loss_2p3_str, ' Time[ms]: ', time_2p2_str, 'batch:' , batch_cnt, '/', train_batch_num, ' ]',end='\r')

    local_acc_y = []
    local_acc_edge = []
    for batch in test_loader:
        pred_y, pred_edge_label = model(
            x = batch.x,
            edge_index = batch.edge_index,
            edge_attr  = batch.edge_attr,
            batch = batch.batch,
            u = None
        )
        pred_y = pred_y.detach().numpy()
        pred_y = np.round(pred_y)

        pred_edge_label = pred_edge_label.detach().numpy()
        pred_edge_label = np.round(pred_edge_label)
        # accuracy is the number of correct predictions divided by the number of predictions
        acc_y = np.sum(pred_y == batch.y.reshape(len(batch.y), 1).numpy()) / len(batch.y)
        acc_edge = np.sum(pred_edge_label == batch.edge_label.reshape(len(batch.edge_label), 1).numpy()) / len(batch.edge_label)
        test_acc_y.append(acc_y)
        local_acc_y.append(acc_y)
        test_acc_edge.append(acc_edge)
        local_acc_edge.append(acc_edge)
    print('\n-        ', epoch, ' Node accuracy: ', str(np.mean(local_acc_y))[:5], ' Edge accuracy: ', str(np.mean(local_acc_edge))[:5])

print('Overallest node accuracy: ', np.mean(test_acc_y))
print('Overallest edge accuracy: ', np.mean(test_acc_edge))
        
figure_train = plt.figure(figsize=(10, 5))
plt.plot(loss_array)
plt.title('Training edge loss')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.savefig('./Node_Edge_Classification/pics/train_loss.png')

# test on test data
figure_test = plt.figure(figsize=(10, 5))
plt.plot(test_acc_y, label='Node accuracy')
plt.plot(test_acc_edge, label='Edge accuracy')

plt.title('Test accuracy')
plt.xlabel('Batch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('./Node_Edge_Classification/pics/test_acc.png')

