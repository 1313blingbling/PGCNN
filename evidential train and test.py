import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import argparse
import numpy as np
from scipy.special import softmax
from polymernet.data import PolymerDataset
from polymernet.model import SingleTaskNet
from torch_geometric.utils import softmax
from sklearn.metrics import mean_squared_error,r2_score
import evidential_deep_learning as edl

parser = argparse.ArgumentParser('Graph Network for polymers')
parser.add_argument('root_dir', help='path to directory that stores data', default ="./data")
parser.add_argument('--split', type=int, default=1,
                   help='CV split that is used for validation (default: 0)')
parser.add_argument('--total-split', type=int, default=10,
                   help='Total number of CV splits (default: 10)')
parser.add_argument('--pred-path', default=None, help='path to prediction csv')
parser.add_argument('--fea-len', type=int, default=16, help='feature length '
                   'for the network (default: 16)')
parser.add_argument('--lr', type=float, default=1e-4, #1e-1
                   help='learning rate (default: 1e-5)')
parser.add_argument('--n-layers', type=int, default=4,
                   help='number of graph convolution layers (default: 4)')
parser.add_argument('--n-h', type=int, default=2,
                   help='number of hidden layers after pool (default: 2)')
parser.add_argument('--epochs', type=int, default=20,
                   help='number of epochs (default: 200)')
parser.add_argument('--batch-size', type=int, default=16,
                   help='batch size (default: 16)')
parser.add_argument('--has-h', type=int, default=0,
                   help='whether to have explicit H (default: 0)')
parser.add_argument('--form-ring', type=int, default=1,
                   help='whether to form ring for molecules (default: 1)')
parser.add_argument('--log10', type=int, default=1,
                   help='whether to use the log10 of the property')
parser.add_argument('--size-limit', type=int, default=None,
                   help='limit the size of training data (default: None)')

result_folder = r"C:\Users\admin\Desktop\PGCNN\results\active"
def write_results(poly_ids,preds,targets,vars,filename):
    if not os.path.exists(result_folder):  
        os.makedirs(result_folder)  
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        for poly_id, pred, target, var in zip(poly_ids, preds, targets, vars):
            writer.writerow([poly_id, pred, target, var]) 

def write_attentions(poly_ids, smiles, attentions, fname):
   with open(fname, 'w') as f:
       writer = csv.writer(f)
       for poly_id, s, a in zip(poly_ids, smiles, attentions):
           writer.writerow([poly_id, s] + a)

def normalization(dataset):
   ys = np.array([data.y for data in dataset])
   return ys.mean(), ys.std()

def get_attention(model, data):
   """Get attention using layers in the model."""
   out = F.leaky_relu(model.node_embed(data.x))
   edge_attr = F.leaky_relu(model.edge_embed(data.edge_attr))

   for cgconv in model.cgconvs:
       out = cgconv(out, data.edge_index, edge_attr)
   size = data.batch[-1].item() + 1
   gate = model.pool.gate_nn(out).view(-1, 1)
   out = model.pool.nn(out) if model.pool.nn is not None else out
   assert gate.dim() == out.dim() and gate.size(0) == out.size(0)
   gate = softmax(gate, data.batch, num_nodes=size)
   gate = gate.squeeze(dim=-1)
   gate = gate.cpu().detach().numpy()
   batch = data.batch.cpu().detach().numpy()
   attentions = [[] for _ in range(size)]
   for g, b in zip(gate, batch):
       attentions[b].append(g)
   return attentions

def main(args):
   has_H, form_ring = bool(args.has_h), bool(args.form_ring)
   log10 = bool(args.log10)
   train_dataset = PolymerDataset(
       args.root_dir, 'train', args.split, form_ring=form_ring, has_H=has_H,
       log10=log10, total_split=args.total_split, size_limit=args.size_limit)
   val_dataset = PolymerDataset(
       args.root_dir, 'val', args.split, form_ring=form_ring, has_H=has_H,
       log10=log10, total_split=args.total_split)
   test_dataset = PolymerDataset(
       args.root_dir, 'test', args.split, form_ring=form_ring, has_H=has_H,
       log10=log10, total_split=args.total_split)
   
   data_example = train_dataset[0]
   train_mean, train_std = normalization(train_dataset)

   train_loader = DataLoader(
       train_dataset, batch_size=args.batch_size, shuffle=True)
   val_loader = DataLoader(
       val_dataset, batch_size=args.batch_size, shuffle=True)
   test_loader = DataLoader(
       test_dataset, batch_size=args.batch_size, shuffle=False)

   pred_dataset = PolymerDataset(  
       args.root_dir, 'pred', args.split, form_ring=form_ring, has_H=has_H,log10=log10,
       total_split=args.total_split)
   pred_loader = DataLoader(  
       pred_dataset,batch_size=args.batch_size, shuffle=False)
   
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
   model = SingleTaskNet(data_example.num_features,
                     data_example.num_edge_features,
                     args.fea_len,
                     args.n_layers,
                     args.n_h,
                     ).to(device)
   
   optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=20, min_lr=1e-5)
   
   def train(epoch):
       model.train()
       loss_all = 0
  
       for data in train_loader:
           data = data.to(device)
           optimizer.zero_grad()
           output = model(data)
           data.y =(data.y - train_mean) / train_std
           data.y = data.y.unsqueeze(1)
           loss = edl.losses.EvidentialRegression(data.y, output, coeff=0.02)
           loss.backward()
           loss_all += loss.item() * data.num_graphs
           optimizer.step()
       return loss_all / len(train_loader.dataset)

   def test(loader):
       model.eval()
       error = 0
       poly_ids = []
       preds = []
       targets = []
       vars = []

       for data in loader:
           data = data.to(device)
           output = model(data)
           output = output.detach().cpu().numpy()
           mu, v, alpha, beta = np.split(output, 4, axis=-1)
           mu = output[:, 0]
           var = np.sqrt(beta / (v * (alpha - 1)))
           var = np.minimum(var, 1e3)[:, 0] 

           pred = mu
           pred = (pred * train_std + train_mean)
           target=data.y.cpu().detach().numpy()
           error += np.abs(pred - target).sum().item() 
           preds.append(pred)
           targets.append(data.y.cpu().detach().numpy())
           vars.append(var)
           poly_ids += data.poly_id

       preds = np.concatenate(preds)
       targets = np.concatenate(targets)
       vars = np.concatenate(vars)
       rmse = np.sqrt(mean_squared_error(targets, preds))
       r2 = r2_score(targets, preds)
       mae= error / len(loader.dataset)
       return mae, poly_ids, preds, targets, rmse, r2, vars    
   
   var_mean = []
   best_val_error = None
   for epoch in range(args.epochs):
      lr = scheduler.optimizer.param_groups[0]['lr']
      loss = train(epoch)
      val_error, poly_ids, preds, targets, val_rmse, val_r2, val_vars = test(val_loader)
      scheduler.step(val_error)
      test_error, poly_ids, preds, targets,  test_rmse, test_r2, test_vars= test(test_loader) 

      if best_val_error is None or val_error <= best_val_error:
         test_error, poly_ids, preds, targets, test_rmse, test_r2, test_vars= test(test_loader)
         best_val_error = val_error
         test_results_path = os.path.join(result_folder, 'active_test_results.csv')
         write_results(poly_ids, preds, targets, test_vars, test_results_path)
         model_save_path = os.path.join(result_folder, 'active_1_model.pth')
         torch.save(model.state_dict(), model_save_path)

      var_mean.append(test_vars.mean())
      test_var_mean=np.mean(test_vars)
      val_var_mean = np.mean(val_vars)
      

      print('Epoch: {:03d}, LR: {:8f}, Loss: {:.7f}, Validation MAE: {:.10f}, '
      'Validation RMSE: {:.10f}, Validation R2: {:.7f}, Validation var: {:.7f}, '
      'Best Validation MAE: {:.10f}, '
      'Test MAE: {:.7f}, Test RMSE: {:.10f}, Test R2: {:.7f}, '
      'Test Var: {:.7f}'.format(
        epoch, lr, loss, val_error, val_rmse, val_r2, val_var_mean,
        best_val_error, test_error, 
        test_rmse, test_r2, test_var_mean))

    # Predict on pred dataset
   model_path = os.path.join(result_folder, 'active_1_model.pth')
   model.load_state_dict(torch.load(model_path))
   model.eval()
   poly_ids = []
   preds = []
   targets = []
   vars = []
   smiles = []
   attentions = []
   for data in pred_loader:
       data = data.to(device)
       output = model(data)
       output = output.detach().cpu().numpy()
       mu, v, alpha, beta = np.split(output, 4, axis=-1)
       mu = output[:, 0]
       var = np.sqrt(beta / (v * (alpha - 1)))
       var = np.minimum(var, 1e3)[:, 0]
       pred = mu
       pred = pred * train_std + train_mean
       preds.append(pred)
       targets.append(data.y.cpu().detach().numpy())
       poly_ids += data.poly_id
       smiles += data.smiles
       vars.append(var)
       attentions += get_attention(model, data)
   
   preds = np.concatenate(preds)
   targets = np.concatenate(targets)
   vars = np.concatenate(vars)
   pred_results_path = os.path.join(result_folder, 'active_pred_results.csv')
   write_results(poly_ids, preds, targets, vars, pred_results_path)
   attentions_path = os.path.join(result_folder, 'active_attentions.csv')
   write_attentions(poly_ids, smiles, attentions, attentions_path)
   
if __name__ == '__main__':
   args = parser.parse_args([r"E:\PGCNN\data\sample\active\8density_active_76test\200data"])
   main(args)