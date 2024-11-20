import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import os
import json
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Fully-Connected Linear layers process 2 images (input and output) 
        # of 15x15 pixels each

        # If 1 layer only
        #self.fc1 = nn.Linear(2 * 15 * 15, 8)   # 8 Structuring Elems to be predicted

        # If 2+ layers
        self.fc1 = nn.Linear(2 * 15 * 15, 2 * 15 * 15)

        # If 2 layers only
        self.fc2 = nn.Linear(2 * 15 * 15, 8)   # 8 Structuring Elems to be predicted

        # If 3 layers
        #self.fc2 = nn.Linear(2 * 15 * 15, 2 * 15 * 15)
        #self.fc3 = nn.Linear(2 * 15 * 15, 8)   # 8 Structuring Elems to be predicted

    def forward(self, x):
        #x = self.dropout1(x)

        # Input and output images are in separate indices of image dimension.
        # x is [batch_size, 2, 15, 15]  where 2 is for input & output images.
        #print(f"x.shape is {x.shape}")

        # Last FC predicts values for each of the 8 SEs

        x = torch.flatten(x, start_dim=1)   # flatten to 1-D array of 2x15x15 (important, see documentation)

        x = self.fc1(x)

        # If 2+ layers
        x = F.relu(x) #see if it is really needed or not, try the code without it
        x = self.fc2(x)

        # If 3 layers
        #x = F.relu(x)
        #x = self.fc3(x)

        # Compute log(probability) for each of the 8 SEs
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = [0 for i in range(8)]  # num correct per SE
    # NOTE: --test-batch-size must be 1 for the below code to work
    num_samples = [0 for i in range(8)]  # num samples per SE
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct[target.item()] += pred.eq(target.view_as(pred)).sum().item()
            num_samples[target.item()] += 1

    num_test_samples = len(test_loader.dataset)
    test_loss /= num_test_samples

    total_correct = sum(correct)

    print('\nTest at epoch {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        epoch, test_loss, total_correct, len(test_loader.dataset),
        100. * total_correct / len(test_loader.dataset)))
    correct_percent = [100 * c / num_samples[i] for i,c in enumerate(correct)]
    correct_percent_str = ["{:.2f}%".format(c) for c in correct_percent]
    print(f"Correct % predictions for SE 1-8: {correct_percent_str}")
    print(f"Num test samples for SE 1-8: {num_samples}\n")


class TwoImageDataset(torch.utils.data.Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = torch.LongTensor(labels)
        self.transform = transforms.Compose([transforms.ToTensor()])
        
    def __getitem__(self, index):
        io_image = self.samples[index]

        input_image_list = io_image["input"]  # 2D list
        input_image_np = np.array(input_image_list)
        #img = Image.fromarray(input_image_np.astype(np.bool), mode="1") # 1-bit
        #input_img_tensor = self.transform(img)  
        input_img_tensor = self.transform(input_image_np)[0]

        output_image_list = io_image["output"]  # 2D list
        output_image_np = np.array(output_image_list)
        output_img_tensor = self.transform(output_image_np)[0]

        # img shape is [15, 15], so input + output stacked is [2, 15, 15]
        x = torch.stack((input_img_tensor, output_img_tensor))

        x = x.to(dtype=torch.float)

        y = self.labels[index]

        #print(f"sample I/O images: {x}")
        
        return x, y
    
    def __len__(self):
        return len(self.samples)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)') #intitial value o
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    """ OLD MNIST CODE
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    train_dataset = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    test_dataset = datasets.MNIST('../data', train=False,
                       transform=transform)
    """

    # Load all images Json files and concat them to single dict
    samples = []
    labels = []
    data_dir = "./IPARC_ChallengeV2/Dataset/One_SE/"
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".json"):
            with open(data_dir + file_name, "r") as file:
                io_image = json.load(file)   # {input: [[..]], output: [[..]]}
                num_samples = len(io_image)
                samples = samples + io_image
            soln_file_name = file_name[:-5] + "_soln.txt"
            with open(data_dir + soln_file_name, "r") as soln_file:
                transf = soln_file.read()  # e.g. Dilation SE1
                # SE number is last char
                se = int(transf[-2])  # last char is \n
                print(f"For file {soln_file_name}, Transformation is {transf[:-1]}, SE is {se}")
                file_labels = [(se-1) for i in range(num_samples)]  # repeat for num_samples
                labels = labels + file_labels

    num_samples = len(samples)
    print(f"Loaded {len(samples)} samples, {len(labels)} labels")

    dataset = TwoImageDataset(samples, labels) #converts the json files to image matrices

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, 
                            [int(0.9 * num_samples), int(0.1 * num_samples)])

    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **train_kwargs) # ** expands the dict. * expands any list.
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, **test_kwargs) #see documentation

    model = Net().to(device) #this is actually 2 lines, model = Net() intitializes the Net class here. model = .to(device) copies the memory of the Net class from the CPU to the GPU if its there.
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr) # see documentation, see other optimizers

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma) #decides the learning rate to use, everytime we update the parameters. gamma decides how much to change the learning rate. step_size decides after how many epochs we change the lr
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader, epoch)
        scheduler.step()

    if args.save_model: 
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
