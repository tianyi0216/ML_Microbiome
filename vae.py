## VAE using PyTorch for data generation

# imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# VAE class
class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()
        
        # encoder
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.bn1 = nn.BatchNorm1d(h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.bn2 = nn.BatchNorm1d(h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        
        # decoder
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.bn3 = nn.BatchNorm1d(h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.bn4 = nn.BatchNorm1d(h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)
        
    def encoder(self, x):
        h = F.relu(self.bn1(self.fc1(x)))
        h = F.relu(self.bn2(self.fc2(h)))
        return self.fc31(h), self.fc32(h)
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def decoder(self, z):
        h = F.relu(self.bn3(self.fc4(z)))
        h = F.relu(self.bn4(self.fc5(h)))
        return torch.sigmoid(self.fc6(h))
    
    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var
    
# loss function
def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

# main
vae = VAE(x_dim=784, h_dim1=512, h_dim2=256, z_dim=2)
optimizer = torch.optim.Adam(vae.parameters())

latent_space = []

epochs = 30

for epoch in range(epochs):
    vae.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = vae(data)
        loss = loss_function(recon_batch, data, mu, log_var)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        latent_space.append(mu.detach().numpy())
        
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    
def generate_samples(n):
    with torch.no_grad():
        z = torch.randn(n, 2)
        sample = vae.decoder(z).detach().numpy()
    return sample


# conditional VAE

class CVAE(nn.Module):
    def __init__(self, x_dim, y_dim, h_dim1, h_dim2, z_dim):
        super(CVAE, self).__init__()
        # encoder
        self.fc1 = nn.Linear(x_dim + y_dim, h_dim1)
        self.bn1 = nn.BatchNorm1d(h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.bn2 = nn.BatchNorm1d(h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder
        self.fc4 = nn.Linear(z_dim + y_dim, h_dim2)
        self.bn3 = nn.BatchNorm1d(h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.bn4 = nn.BatchNorm1d(h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)
        
    def encoder(self, x, y):
        # Concatenate x and y
        h = torch.cat([x, y], dim=1)
        h = F.relu(self.bn1(self.fc1(h)))
        h = F.relu(self.bn2(self.fc2(h)))
        return self.fc31(h), self.fc32(h)
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def decoder(self, z, y):
        # Concatenate z and y
        h = torch.cat([z, y], dim=1)
        h = F.relu(self.bn3(self.fc4(h)))
        h = F.relu(self.bn4(self.fc5(h)))
        return torch.sigmoid(self.fc6(h))
    
    def forward(self, x, y):
        mu, log_var = self.encoder(x, y)
        z = self.sampling(mu, log_var)
        return self.decoder(z, y), mu, log_var

# You need to adjust dimensions accordingly
# x_dim: Dimension of input data
# y_dim: Dimension of one-hot encoded class labels
# h_dim1, h_dim2: Dimensions of hidden layers
# z_dim: Dimension of the latent space

# Example usage
cvae = CVAE(x_dim=784, y_dim=10, h_dim1=512, h_dim2=256, z_dim=2)

    