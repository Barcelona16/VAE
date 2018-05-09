import argparse
from utils import *

parser = argparse.ArgumentParser(description='PyTorch VAE Training')
parser.add_argument('--data', default='pokemon', help='Dataset name')
parser.add_argument('--epochs', '-e', default=5000, type=int, help='Total epochs to run (default: 5000)')
parser.add_argument('--batch_size', '-bs', default=256, type=int, help='Mini-batch size (default: 256)')
parser.add_argument('--learn_rate', '-lr', default=1e-3, type=int, help='Learning rate (default: 1e-3)')
parser.add_argument('--label', '-l', default='VAE', help='Experiment name', type=str)
parser.add_argument('--checkpoint', '-cp', default=None, type=str, help='Checkpoint name')
  

def main():
    args = parser.parse_args()
    
    gen_data_list()
    
    if not os.path.isdir('data'):
        os.system('mkdir data')
        
    if not os.path.isdir('checkpoints'):
        os.system('mkdir checkpoints')
    
    try:
        net, epoch, losses, bces, kls, optimizer, scheduler = load_checkpoint("./checkpoints/" + args.checkpoint, args.learn_rate)
    except:
        net = Net() # Initialize model
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                net = torch.nn.DataParallel(net)
            net = net.cuda() 
        epoch = 0
        losses = []
        bces = []
        kls = []
        optimizer = optim.Adam(net.parameters(), lr=args.learn_rate, amsgrad=True)
        scheduler = SGDRScheduler(optimizer, min_lr=1e-5, max_lr=args.learn_rate, cycle_length=500, current_step=0)
        print("Starting new training")

    multiSet = MultiSet(args.data)
    dataloader = Utils.DataLoader(dataset=multiSet, shuffle=True, batch_size=args.batch_size)
    
    train_losses, bces, kls = train(net, optimizer, scheduler, dataloader, epoch, args.label, losses, bces, kls, args.epochs)
    generate_animation("data/", args.label)
    print("Training completed!")

    
if __name__ == '__main__':
    main()