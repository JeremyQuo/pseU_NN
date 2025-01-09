import argparse
import pandas as pd
import numpy as np
import os
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from model import GraphCNN
from tqdm import tqdm
import torch

BASES = ['A', 'U', 'C', 'G']
BASE_TO_INDEX = {base: idx for idx, base in enumerate(BASES)}
SEQUENCE_LENGTH = 41


def read_file(directory_path,file_number):
    file_names = os.listdir(directory_path)
    data_list = []
    num_file = 0
    for file_name in file_names:
        if file_number is not None:
            if num_file >= file_number:
                break
        # Read file
        file_path = os.path.join(directory_path, file_name)
        df = pd.read_csv(file_path, skiprows=1, header=None, sep='\t')
        df[0] = df[0] - 1
        df[2] = df[2] - 1

        # Create adjacency matrix
        num_nodes = df.shape[0]
        adj_matrix = np.zeros((num_nodes, num_nodes))

        # Fill adjacency matrix
        for idx, row in df.iterrows():
            u, v = row[0], row[2]
            if v != -1:
                adj_matrix[u][v] = 1
                adj_matrix[v][u] = 1
            if u < num_nodes - 1:
                adj_matrix[u][u + 1] = 1
                adj_matrix[u + 1][u] = 1
            adj_matrix[u][u] = 1

        # Create sequence features
        sequence = "".join(df[1].values.reshape(-1).tolist())
        one_hot = torch.zeros((SEQUENCE_LENGTH, len(BASES)))
        for i, base in enumerate(sequence):
            if base in BASE_TO_INDEX:
                one_hot[i, BASE_TO_INDEX[base]] = 1.0

        # Combine features
        adj_matrix = np.hstack([adj_matrix, one_hot.numpy()])

        # Convert to tensors and move to device
        x = torch.tensor(adj_matrix, dtype=torch.float32).unsqueeze(0)

        # Create Data object
        data = Data(x=x)
        data_list.append(data)
        num_file += 1
    return data_list,file_names[:num_file]

def main(input_file, model_file, output_file, batch_size, device,file_number):
    input_file,file_names = read_file(input_file,file_number)
    model = torch.load(model_file, map_location=torch.device(device))
    test_dataset = DataLoader(input_file, batch_size=batch_size, shuffle=True)
    batch_number = len(input_file)//batch_size + 1
    print('Start to predict ...')
    pbar = tqdm(range(batch_number),unit='batch')
    with torch.no_grad():
        result = torch.tensor([])
        for batch in test_dataset:
            batch = batch.to(device)
            out = model(batch.x, batch.batch_size)
            result = torch.cat((result, out), dim=0)
            pbar.update(1)
    pbar.close()
    df = pd.DataFrame(result.numpy(),columns=['predictions'])
    df['seq_name'] = file_names
    df = df[['seq_name','predictions']]
    if output_file.split('.')[-1]!='csv':
        output_file = output_file+'.csv'
    df.to_csv(output_file,index=False)
    print('Saved as '+ output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-i','--input_folder', type=str, help='Directory path to input files')
    parser.add_argument('--model', type=str, help='Path to the model file',default='model.pth')
    parser.add_argument('-o','--output', type=str, help='Output csv path',default='output.csv')
    parser.add_argument('--batch_size', type=int, help='batch_size of input feature', default=500)
    parser.add_argument('--subsample_number', type=int, help='Subsample number if required', default=None)
    parser.add_argument('--device', type=str, help='Device name', default='cpu')
    args = parser.parse_args()
    main(args.input_folder, args.model, args.output, args.batch_size, args.device, args.subsample_number)



