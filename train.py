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



    
try:
    net, epoch, losses, bces, kls, optimizer = load_checkpoint("./checkpoints/")
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
    optimizer = optim.Adam(net.parameters(), lr=0.001, amsgrad=True)
    scheduler = SGDRScheduler(optimizer, min_lr=1e-5, max_lr=1e-3, cycle_length=500, current_step=0)
    print("Starting new training")



max_epochs = 5000

multiSet = MultiSet('pokemon')
dataLoader = Utils.DataLoader(dataset=multiSet, shuffle=True, batch_size=BATCH_SIZE)

train_losses, bces, kls = train(net, optimizer, scheduler, dataLoader, epoch, "POKEVAE", losses, bces, kls, max_epochs)
