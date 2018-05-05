from model import *


class MultiSet(Utils.Dataset):
    """
    Dataloader for the model. Can easily add more datasets.
    """
    def __init__(self, name='pokemon'):
        if name == 'pokemon':
            self.list = pd.read_csv("pokelist",header=None,delimiter=',').values[0]
    
    def __len__(self):
        return len(self.list)
    
    def __getitem__(self, index):
        data = mpimg.imread(self.list[index])
        data = cv2.resize(data, (RESIZE,RESIZE))/255
        return data

    
def gen_data_list():
    if not os.path.isfile('pokelist'):
        if os.path.exists('./Pokemon') and os.path.exists('./PokemonFlip'):
            with open("pokelist", 'w') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                data = [os.path.abspath("./Pokemon/")+"/"+file for file in os.listdir("./Pokemon/")]
                data += [os.path.abspath("./PokemonFlip/")+"/"+file for file in os.listdir("./PokemonFlip/")]
                wr.writerow(data)
            print("Generated list of Pokemon image paths, both normal and flipped")
        elif os.path.exists('./Pokemon'):
            with open("pokelist", 'w') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                data = [os.path.abspath("./Pokemon/")+"/"+file for file in os.listdir("./Pokemon/")]
                wr.writerow(data)
            print("Generated list of Pokemon image paths")
        else:
            print("Missing Pokemon folder with images")
    else:
        print("Pokemon image list available")
    

def load_checkpoint(filename, LR):
    '''
    Loading function for the model before and during training
    From a checkpoint file, it loads and returns all necessary data (mode, optimiser, epoch number, losses)
    input: filename -> The name of the checkpoint file to be opened (.pth or .pt)
    output: net -> The saved model, including weights and biases
    output: epoch -> The epoch number at which the training was saved
    output: loss_save -> An array of all the saved batch losses during training
    output: optimizer -> The current state of the optimiser with its updated learning rates from training
    '''
    
    net = Net() # Initialize model 
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            net = torch.nn.DataParallel(net)
        net = net.cuda()
    checkpoint = torch.load(filename) # load checkpoint data
    net.load_state_dict(checkpoint['state_dict'])
    epoch = checkpoint['epoch']
    losses = checkpoint['losses']
    bces = checkpoint['bces']
    kls = checkpoint['kls']
    cs = checkpoint['cs']
    optimizer = optim.Adam(net.parameters(), lr=LR, amsgrad=True)
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler = SGDRScheduler(optimizer, min_lr=1e-5, max_lr=LR, cycle_length=500, current_step=cs)
    
    print("Loaded checkpoint:", filename)
    return net, epoch, losses, bces, kls, optimizer, scheduler


def multi_plot(images, model, ROW=4, COL=4):
    """
    To plot an array of images
    Need batch size of row*col and a screen
    input: batch of image arrays
    """
    try:
        f, axarr = plt.subplots(ROW, COL, figsize=(15, ROW*4))
        for row in range(ROW//2):
            for col in range(COL):
                image = images[col+(COL*row),:,:,:].unsqueeze(0)
                axarr[2*row,col].imshow(image.squeeze().numpy())
                image = image.permute(0,3,1,2)
                x_out, z_mean, z_logvar = model(Variable(image.float().cuda()))
                x_out = x_out.permute(0,2,3,1)
                axarr[2*row+1,col].imshow(x_out.data.cpu().squeeze().numpy())
        plt.show()
    except:
        pass


def criterion(x_out, target, z_mean, z_logvar, alpha=1, beta=20):
    """
    Criterion for VAE done analytically
    output: loss
    output: bce
    output: KL Divergence
    """
    bce = F.mse_loss(x_out, target, size_average=False) #Use MSE loss for images
    kl = -0.5 * torch.sum(1 + z_logvar - (z_mean**2) - torch.exp(z_logvar)) #Analytical KL Divergence - Assumes p(z) is Gaussian Distribution
    loss = ((alpha * bce) + (beta * kl)) / x_out.size(0)    
    return loss, bce, kl


def train(model, optimizer, scheduler, dataloader, epoch, label, losses, bces, kls, max_epochs):
    step = 0
    for _ in range(max_epochs):
        for images in dataloader:
            optimizer.zero_grad()
            
            image_in = images.permute(0,3,1,2)
            x_in = Variable(image_in.float().cuda())
            x_out, z_mu, z_logvar = model(x_in)
            loss, bce, kl = criterion(x_out, x_in, z_mu, z_logvar)
            loss.backward()
            scheduler.step()
            optimizer.step()
            losses.append(loss.item())
            bces.append(bce.item())
            kls.append(kl.item())
            
            if epoch%10 == 0 and epoch != 0:
                save_file = "checkpoints/" + label + "_epoch_{:06d}".format(epoch) + '.pth'
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'losses' : losses,
                    'bces' : bces,
                    'kls' : kls,
                    'cs' : step
                }, save_file)
                print("Saved checkpoint")
                
            step += 1
        epoch += 1
        clear_output(wait=True)
        print("Epoch:", epoch, '- Loss: {:3f}'.format(loss.item()))
        multi_plot(images, model)
    return losses, bces, kls


### The following was taken from https://github.com/A-Jacobson/tacotron2

def adjust_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

        
class SGDRScheduler:
    """
    Implements STOCHASTIC GRADIENT DESCENT WITH WARM RESTARTS (SGDR)
    with cosine annealing from https://arxiv.org/pdf/1608.03983.pdf.
    #TODO add cycle mult and min, max lr annealing.
    """

    def __init__(self, optimizer, min_lr, max_lr, cycle_length, warmup_steps=5, current_step=0):
        self.optimizer = optimizer
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr = optimizer.param_groups[0]['lr']
        self.cycle_length = cycle_length
        self.current_step = current_step
        self.warmup_steps = warmup_steps

    def calculate_lr(self):
        """
        calculates new learning rate with cosine annealing
        """
        step = self.current_step % self.cycle_length  # get step in current cycle
        self.lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * \
                  (1 + np.cos((step / self.cycle_length) * np.pi))

    def step(self):
        self.current_step += 1
        self.calculate_lr()
        if self.current_step in range(self.warmup_steps):
            self.lr /= 10.0  # take a few steps with a lower lr to "warmup"
        adjust_lr(self.optimizer, self.lr)