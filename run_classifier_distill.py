
import os
import argparse
import dataset
import model.backbone as backbone
import metric.pairsampler as pair
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets.cifar import CIFAR100 as cifar100
from torchvision.datasets.imagenet import ImageNet as imagenet
from tqdm import tqdm
from torch.utils.data import DataLoader
from metric.loss import HardDarkRank, RkdDistance, RKdAngle, L2Triplet, AttentionTransfer, HKD,NST
from model.embedding import LinearEmbedding


parser = argparse.ArgumentParser()
LookupChoices = type('', (argparse.Action, ), dict(__call__=lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))
###@xy###
parser.add_argument('--dataset',
                    choices=dict(cifar100=cifar100,
                                 imagenet=imagenet,
                                 dog=dataset.DogsDataset
                                 ),
                    default=cifar100,
                    action=LookupChoices)

parser.add_argument('--base',
                    choices=dict(resnet18=backbone.ResNet18,
                                 resnet50=backbone.ResNet50,
                                 shufflenetv2=backbone.shufflenetv2,
                                 renet18_cifar=backbone.ResNet18_cifar),
                    default=backbone.ResNet18_cifar,
                    action=LookupChoices)

parser.add_argument('--teacher_base',
                    choices=dict(resnet18=backbone.ResNet18,
                                 resnet50=backbone.ResNet50,
                                 resnet152=backbone.ResNet152,
                                 resnet50_cifar=backbone.ResNet50_cifar),
                    default=backbone.ResNet50_cifar,
                    action=LookupChoices)

parser.add_argument('--triplet_ratio', default=0, type=float)
parser.add_argument('--dist_ratio', default=0, type=float)
parser.add_argument('--angle_ratio', default=0, type=float)
parser.add_argument('--dark_ratio', default=0, type=float)
parser.add_argument('--dark_alpha', default=1, type=float)
parser.add_argument('--dark_beta', default=1, type=float)
parser.add_argument('--at_ratio', default=0, type=float)
parser.add_argument('--hkd_ratio',default=0, type=float)
parser.add_argument('--nst_ratio', default=0, type=float)
parser.add_argument('--ground_truth_ratio', default=0, type=float)

parser.add_argument('--triplet_sample',
                    choices=dict(random=pair.RandomNegative,
                                 hard=pair.HardNegative,
                                 all=pair.AllPairs,
                                 semihard=pair.SemiHardNegative,
                                 distance=pair.DistanceWeighted),
                    default=pair.DistanceWeighted,
                    action=LookupChoices)

parser.add_argument('--triplet_margin', type=float, default=0.2)
parser.add_argument('--l2normalize', choices=['true', 'false'], default='false')
###@xy###
parser.add_argument('--embedding', default=False, type=bool)
parser.add_argument('--embedding_size', default=200, type=int)
parser.add_argument('--extract_feature_mode',default=False,type=bool)

parser.add_argument('--teacher_load', default='cifar100_teacher_model/best.pth')
parser.add_argument('--teacher_l2normalize', choices=['true', 'false'], default='false')
###@xy###
parser.add_argument('--teacher_embedding', default=False, type=bool)
parser.add_argument('--teacher_embedding_size', default=200, type=int)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--data', default='data')
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batch', default=128, type=int)
parser.add_argument('--iter_per_epoch', default=100, type=int)
parser.add_argument('--lr_decay_epochs', type=int, default=[40,80,100], nargs='+')
parser.add_argument('--lr_decay_gamma', type=float, default=0.1)
parser.add_argument('--save_dir', default='dog_student')
parser.add_argument('--load', default=None)
opts = parser.parse_args()
#argument setting for dogs
opts = parser.parse_args(['--dataset','dog','--base','resnet18','--embedding','True','--embedding_size','120',
                          '--teacher_embedding','True','--teacher_embedding_size','120',
                          '--teacher_base', 'resnet152', '--teacher_load', 'teacher_model/best.pth',
                        '--dist_ratio','50', '--angle_ratio', '50', '--ground_truth_ratio','0.5',
                          '--nst_ratio','0','--dark_ratio','0',
                         '--lr','0.001', '--hkd_ratio', '50','--at_ratio','0','--batch','128', '--epochs','200',
                          '--save_dir', 'dog_student_resnet50_resnet18'])
student_base = opts.base()
teacher_base = opts.teacher_base()


def get_normalize(net):
    # you can add Customized normalize with different type of your net model
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1).cuda()
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1).cuda()
    def norm(x):
        x = (x - mean) / std
        return x
    return norm


teacher_normalize = get_normalize(teacher_base)
student_normalize = get_normalize(student_base)
###@xy###
#you may need to modify these transforms with your dataset
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

dataset_train = opts.dataset(opts.data, train=True, transform=train_transform, download=True)
dataset_train_eval = opts.dataset(opts.data, train=False, transform=test_transform, download=False)
dataset_eval = opts.dataset(opts.data, train=False, transform=test_transform, download=False)


print("Number of images in Training Set: %d" % len(dataset_train))
print("Number of images in Test set: %d" % len(dataset_eval))
###@xy###
loader_train_sample = DataLoader(dataset_train,batch_size=opts.batch, shuffle=True,num_workers=8)
loader_train_eval = DataLoader(dataset_train_eval, shuffle=False, batch_size=opts.batch, drop_last=False,
                               pin_memory=False, num_workers=8)
loader_eval = DataLoader(dataset_eval, shuffle=False, batch_size=opts.batch, drop_last=False,
                               pin_memory=True, num_workers=8)
#based on based_model,add a embeding layer to adapt your class number
if opts.embedding:
    print('student embedding')
    student = LinearEmbedding(student_base,
                          output_size=student_base.output_size,
                          embedding_size=opts.embedding_size,
                          normalize=opts.l2normalize == 'true')
else:
    student=student_base

if opts.load is not None:
    student.load_state_dict(torch.load(opts.load))
    print("Loaded studnet Model from %s" % opts.load)
if opts.teacher_embedding:
    print('teacher embedding')
    teacher = LinearEmbedding(teacher_base,
                          output_size=teacher_base.output_size,
                          embedding_size=opts.teacher_embedding_size,
                          normalize=opts.teacher_l2normalize == 'true')
else: teacher=teacher_base
# only train the embedding layer
if opts.extract_feature_mode:
    for param in student.parameters()[:-1]:
        param.requires_grad = False

teacher.load_state_dict(torch.load(opts.teacher_load))
student = student.cuda()
teacher = teacher.cuda()
###@xy###
#optimizer = optim.Adam(student.parameters(), lr=opts.lr, weight_decay=1e-5)
optimizer = optim.SGD(student.parameters(),lr=opts.lr,momentum=0.9,weight_decay=5e-4)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opts.lr_decay_epochs, gamma=opts.lr_decay_gamma)

dist_criterion = RkdDistance()
angle_criterion = RKdAngle()
dark_criterion = HardDarkRank(alpha=opts.dark_alpha, beta=opts.dark_beta)
at_criterion = AttentionTransfer()
hkd_criterion = HKD()
nst_criterion = NST()
crossentropy_criterion=torch.nn.CrossEntropyLoss()
def train(loader, ep):
    lr_scheduler.step()
    student.train()
    teacher.eval()

    dist_loss_all = []
    angle_loss_all = []
    dark_loss_all = []
    at_loss_all = []
    loss_all = []
    hkd_loss_all = []
    ground_loss_all=[]
    nst_loss_all=[]

    train_iter = tqdm(loader)
    for images, labels in train_iter:
        images, labels = images.cuda(), labels.cuda()
###@xy###
        with torch.no_grad():
            t_b1, t_b2, t_b3, t_b4, pool, t_e = teacher(teacher_normalize(images), True)

        b1, b2, b3, b4, pool, e = student(student_normalize(images), True)
        at_loss = opts.at_ratio * (at_criterion(b2, t_b2) + at_criterion(b3, t_b3) + at_criterion(b4, t_b4))
        nst_loss=opts.nst_ratio * (nst_criterion(b2,t_b2) + nst_criterion(b3, t_b3) + nst_criterion(b4, t_b4))
###@xy###
        dist_loss = opts.dist_ratio * dist_criterion(e, t_e)
        angle_loss = opts.angle_ratio * angle_criterion(e, t_e)
        dark_loss = opts.dark_ratio * dark_criterion(e, t_e)
        hkd_loss = opts.hkd_ratio * hkd_criterion(e,t_e, 4)
        ground_loss=opts.ground_truth_ratio*crossentropy_criterion(e, labels)
        loss = dist_loss + angle_loss + dark_loss + at_loss+ground_loss+hkd_loss+nst_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        dist_loss_all.append(dist_loss.item())
        angle_loss_all.append(angle_loss.item())
        dark_loss_all.append(dark_loss.item())
        at_loss_all.append(at_loss.item())
        ground_loss_all.append(ground_loss.item())
        hkd_loss_all.append(hkd_loss.item())
        loss_all.append(loss.item())
        nst_loss_all.append(nst_loss.item())
###@xy###
    print('[Epoch %d] Loss: %.5f, Dist: %.5f, Angle: %.5f, Dark: %.5f ,At: %.5f, HKD: %.5f, NST:%.5f,ground_loss: %.5f\n' %\
          (ep, torch.Tensor(loss_all).mean(),
           torch.Tensor(dist_loss_all).mean(), torch.Tensor(angle_loss_all).mean(), torch.Tensor(dark_loss_all).mean(),
           torch.Tensor(at_loss_all).mean(), torch.Tensor(hkd_loss_all).mean(), torch.Tensor(nst_loss_all).mean(),
           torch.Tensor(ground_loss_all).mean()
           ))
###@xy###
def eval(net, normalize,loader, ep):
    net.eval()
    test_iter = tqdm(loader, ncols=80)
    embeddings_all, labels_all = [], []
    test_iter.set_description("[Eval][Epoch%d]" %ep)
    with torch.no_grad():
        for images, labels in test_iter:
            images, labels = images.cuda(), labels.cuda()
            pool,embedding = net(normalize(images),False)
            embeddings_all.append(embedding.data)
            labels_all.append(labels.data)
        embeddings_all = torch.cat(embeddings_all).cuda()

        labels_all = torch.cat(labels_all).cuda()
        predicted = torch.argmax(embeddings_all,1)
        correct = (predicted == labels_all).sum()
        accuracy = (100*correct).float()/(labels_all.size(0))
        print('[Epoch %d] Test Accuracy:[%.2f]'%(ep,accuracy),"%")
    return accuracy.item()
eval(teacher, teacher_normalize, loader_eval, 0)
###@xy###
best_val_acc = eval(student, student_normalize, loader_eval,0)
for epoch in range(1, opts.epochs+1):
    train(loader_train_sample, epoch)
    val_acc = eval(student, student_normalize, loader_eval, epoch)
    if best_val_acc < val_acc:
        best_val_acc = val_acc
        if opts.save_dir is not None:
            if not os.path.isdir(opts.save_dir):
                os.mkdir(opts.save_dir)
            torch.save(student.state_dict(), "%s/%s" % (opts.save_dir, "best.pth"))

    if opts.save_dir is not None:
        if not os.path.isdir(opts.save_dir):
            os.mkdir(opts.save_dir)
        torch.save(student.state_dict(), "%s/%s" % (opts.save_dir, "last.pth"))
        with open("%s/result.txt" % opts.save_dir, 'w') as f:
           # f.write('Best Train Accuracy: %.4f\n' % (best_train_acc * 100))
            f.write("Best Test Accuracy: %.4f\n" % (best_val_acc * 100))
            f.write("Final Accuracy: %.4f\n" % (val_acc * 100))
    print("Best Eval Accuracy: %.4f" % best_val_acc)
