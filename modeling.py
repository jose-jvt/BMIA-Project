import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, DataLoader
from torch import nn
from gat import GATRegressor
from dataset import BrainGraphDataset

def matrix_to_graph(matrix, threshold=0.5) -> Data:
    edge_index = []
    edge_attr = []

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i != j and abs(matrix[i, j]) > threshold:
                edge_index.append([i, j])
                edge_attr.append(matrix[i, j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # [2, num_edges]
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    data = Data(edge_index=edge_index, edge_attr=edge_attr)

    return data

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)

def test(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            # TODO: Implement eval metric
    # TODO: Return avg metric
    return 1

def pipeline():
    # Assume dataset is already created
    your_labels_dict = ... # TODO: Load or create your labels dictionary
    dataset = BrainGraphDataset(mat_dir='path/to/mat_files', labels_dict=your_labels_dict)

    # Create indices for train, val, test
    indices = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=42)

    # Use index slicing to split dataset
    train_dataset = dataset[train_idx]
    val_dataset = dataset[val_idx]
    test_dataset = dataset[test_idx]

    # DataLoaders
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Model, optimizer, and loss function
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GATRegressor(in_channels=246, hidden_channels=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(1, 101):
        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}')

    test_eval = test(model, test_loader, device)
    print(f'Test Evaluation Metric: {test_eval:.4f}')

