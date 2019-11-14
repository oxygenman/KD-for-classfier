import os
import argparse
import random
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets.cifar import CIFAR100 as cifar100
from torchvision.datasets.imagenet import ImageNet as imagenet
import dataset
import model.backbone as backbone
import metric.loss as loss
import metric.pairsampler as pair
from tqdm import tqdm
from torch.utils.data import DataLoader
from model.embedding import LinearEmbedding

parser = argparse.ArgumentParser()
LookupChoices = type('', (argparse.Action, ), dict(__call__=lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))
print(dict(__call__=lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))

parser.add_argument('--mode',
                    choices=["train", "eval"],
                    default="train")

parser.add_argument('--load',
                    default=None)

parser.add_argument('--dataset',
                    choices=dict(cub200=dataset.CUB2011Metric,
                                 cars196=dataset.Cars196Metric,
                                 stanford=dataset.StanfordOnlineProductsMetric,
                                 imagenet = imagenet,
                                 cifar100=cifar100,
                                 Dogs=dataset.DogsDataset),
                    default=cifar100,
                    action=LookupChoices)

parser.add_argument('--base',
                    choices=dict(googlenet=backbone.GoogleNet,
                                 inception_v1bn=backbone.InceptionV1BN,
                                 resnet18=backbone.ResNet18,
                                 resnet50=backbone.ResNet50,
                                 resnet152=backbone.ResNet152,
                                 shufflenetv2=backbone.ShuffleNet,
                                 resnet50_cifar=backbone.ResNet50_cifar,
                                 resnet18_cifar=backbone.ResNet18_cifar),
                    default=backbone.ResNet18,
                    action=LookupChoices)

parser.add_argument('--sample',
                    choices=dict(random=pair.RandomNegative,
                                 hard=pair.HardNegative,
                                 all=pair.AllPairs,
                                 semihard=pair.SemiHardNegative,
                                 distance=pair.DistanceWeighted),
                    default=pair.AllPairs,
                    action=LookupChoices)

parser.add_argument('--loss',
                    choices=dict(l1_triplet=loss.L1Triplet,
                                 l2_triplet=loss.L2Triplet,
                                 contrastive=loss.ContrastiveLoss),
                    default=loss.L2Triplet,
                    action=LookupChoices)

parser.add_argument('--margin', type=float, default=0.2)
parser.add_argument('--add_embedding',type=bool, default=False)
parser.add_argument('--embedding_size', type=int, default=0)
parser.add_argument('--l2normalize', choices=['true', 'false'], default='false')
parser.add_argument('--extract_feature_mode', type=bool, default=False)

parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--lr_decay_epochs', type=int, default=[200], nargs='+')
parser.add_argument('--lr_decay_gamma', default=0.1, type=float)

parser.add_argument('--batch', default=32, type=int)
parser.add_argument('--num_image_per_class', default=5, type=int)

parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--iter_per_epoch', type=int, default=100)
parser.add_argument('--recall', default=[1], type=int, nargs='+')

parser.add_argument('--seed', default=random.randint(1, 1000), type=int)
parser.add_argument('--data', default='/home/xy/pan/data')
parser.add_argument('--save_dir', default='resnet50_base_model')
#@xy set the arguments
#Dogs Dataset setting
opts = parser.parse_args(['--mode','train','--dataset','Dogs', '--base', 'resnet18', '--lr','0.001',
                          '--add_embedding','True','--embedding_size','120','--batch','64',
                          '--epoch','200','--save_dir','dog_resnet18_model'])
#Cifar-100 Dataset setting
#opts = parser.parse_args(['--mode','eval','--dataset','cifar-100','--base','resnet50',
#                      '--add_embedding', 'True', '--embedding_size','100'])
#ImageNet Dataset setting
#opts = parser.parse_args(['--mode','eval','--dataset','imagenet','--base','resnet50',
#                        '--add_embedding', 'False'])

for set_random_seed in [random.seed, torch.manual_seed, torch.cuda.manual_seed_all]:
    set_random_seed(opts.seed)

base_model = opts.base(pretrained=True)
if isinstance(base_model, backbone.InceptionV1BN) or isinstance(base_model, backbone.GoogleNet):
    normalize = transforms.Compose([
        transforms.Lambda(lambda x: x[[2, 1, 0], ...] * 255.0),
        transforms.Normalize(mean=[104, 117, 128], std=[1, 1, 1]),
    ])
else:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

###@xy###
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    #transforms.Pad(4),
    #transforms.RandomCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

###@xy###
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])
if opts.dataset==imagenet:
    print('imagenet without embedding')
    dataset_train = opts.dataset(opts.data, split='train', transform=train_transform, download=False)
    dataset_train_eval = opts.dataset(opts.data, split='train', transform=test_transform, download=False)
    dataset_eval = opts.dataset(opts.data, split='val', transform=test_transform, download=False)
else:
    dataset_train = opts.dataset(opts.data, train=True, transform=train_transform, download=False)
    dataset_train_eval = opts.dataset(opts.data, train=True, transform=test_transform, download=False)
    dataset_eval = opts.dataset(opts.data, train=False, transform=test_transform, download=False)

print("Number of images in Training Set: %d" % len(dataset_train))
print("Number of images in Test set: %d" % len(dataset_eval))

loader_train_sample = DataLoader(dataset_train,batch_size=opts.batch, shuffle=True,num_workers=8)
loader_train_eval = DataLoader(dataset_train_eval, shuffle=False, batch_size=opts.batch, drop_last=False,
                                                                 pin_memory=False, num_workers=8)
loader_eval = DataLoader(dataset_eval, shuffle=False, batch_size=opts.batch, drop_last=False,
                                                                 pin_memory=False, num_workers=8)
#@xy Wheather to add a embedding layer with random initial
if opts.add_embedding:
    print('add embedding layer')
    model = LinearEmbedding(base_model,
                        output_size=base_model.output_size,
                        embedding_size=opts.embedding_size,
                        normalize=opts.l2normalize == 'true').cuda()
else:
    model=base_model.cuda()

#@xy Wheather to freeze the layers before the last layer
if opts.extract_feature_mode:
    for param in model.parameters()[:-1]:
        param.requires_grad = False
if opts.load is not None:
    model.load_state_dict(torch.load(opts.load))
    print("Loaded Model from %s" % opts.load)

criterion = torch.nn.CrossEntropyLoss()
#optimizer = optim.Adam(model.parameters(),lr=opts.lr,weight_decay=1e-5)
optimizer = optim.SGD(model.parameters(),lr=opts.lr,momentum=0.9,weight_decay=1e-4)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opts.lr_decay_epochs, gamma=opts.lr_decay_gamma)

def train(net, loader, ep):
    lr_scheduler.step()
    net.train()
    loss_all, norm_all = [], []
    train_iter = tqdm(loader, ncols=80)
    for images, labels in train_iter:
        images, labels = images.cuda(), labels.cuda()
        pool,embedding = net(images)
        loss = criterion(embedding, labels)
        loss_all.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_iter.set_description("[Train][Epoch %d] Loss: %.5f" % (ep, loss.item()))
    print('[Epoch %d] MeanLoss: %.5f\n' % (ep, torch.Tensor(loss_all).mean()))

def eval(net, loader, ep):
    net.eval()
    test_iter = tqdm(loader, ncols=80)
    embeddings_all, labels_all = [], []
    test_iter.set_description("[Eval][Epoch%d]" %ep)
    with torch.no_grad():
        for images, labels in test_iter:
            images, labels = images.cuda(), labels.cuda()
            pool,embedding = net(images,False)
            embeddings_all.append(embedding.data)
            labels_all.append(labels.data)
        embeddings_all = torch.cat(embeddings_all).cuda()
        labels_all = torch.cat(labels_all).cuda()
        predicted = torch.argmax(embeddings_all,1)
        correct = (predicted == labels_all).sum()
        accuracy = (100*correct).float()/(labels_all.size(0))
        print('[Epoch %d] Accuracy:[%.2f]'%(ep,accuracy))
    return accuracy.item()
if opts.mode == "eval":
    #eval(model,loader_train_eval,0)
    eval(model,loader_eval,0)
else:
    #train_acc = eval(model, loader_train_eval, 0)
    val_acc = eval(model, loader_eval, 0)
    best_acc = val_acc

    for epoch in range(1, opts.epochs+1):
        train(model,loader_train_sample, epoch)
        #train_acc = eval(model, loader_train_eval, epoch)
        val_acc = eval(model, loader_eval, epoch)
        if best_acc < val_acc:
            best_acc = val_acc
            if opts.save_dir is not None:
                if not os.path.isdir(opts.save_dir):
                    os.mkdir(opts.save_dir)
                torch.save(model.state_dict(), "%s/%s"%(opts.save_dir, "best.pth"))
        if opts.save_dir is not None:
            if not os.path.isdir(opts.save_dir):
                os.mkdir(opts.save_dir)
            torch.save(model.state_dict(),"%s/%s"%(opts.save_dir, "last.pth"))
            with open("%s/result.txt" % opts.save_dir, 'w') as f:
                f.write("Best Accuracy: %.4f\n" % (best_acc * 100))
                f.write("Final Accuracy: %.4f\n" % (val_acc * 100))
            print("Best Accuracy: %.4f" % best_acc)
