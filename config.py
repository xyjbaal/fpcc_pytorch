import argparse


# Training settings
parser = argparse.ArgumentParser(description='Point Cloud Recognition')
parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                    help='Name of the experiment')
parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                    choices=['pointnet', 'dgcnn'],
                    help='Model to use, [pointnet, dgcnn]')
parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                    choices=['modelnet40'])
parser.add_argument('--batch_size', type=int, default=4, metavar='batch_size',
                    help='Size of batch)')
parser.add_argument('--epochs', type=int, default=250, metavar='N',
                    help='number of episode to train ')
parser.add_argument('--use_sgd', type=bool, default=False,
                    help='Use SGD')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001, 0.1 if using sgd)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                    choices=['cos', 'step'],
                    help='Scheduler to use, [cos, step]')
parser.add_argument('--no_cuda', type=bool, default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--eval', type=bool, default=False,
                    help='evaluate the model')
parser.add_argument('--num_points', type=int, default=4096,
                    help='num of points to use')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='initial dropout rate')
parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                    help='Dimension of embeddings')
parser.add_argument('--k', type=int, default=20, metavar='N',
                    help='Num of nearest neighbors to use')
parser.add_argument('--model_path', type=str, default='', metavar='N',
                    help='Pretrained model path')
parser.add_argument('--model_file', type=str, default='', metavar='N',
                    help='Pretrained model file')
parser.add_argument('--group_num', type=int, default=50, help='Maximum Group Number in one pc')
parser.add_argument('--d_max', type=float, default=0.18, help=' ring: 0.18 / gear: 0.25')
parser.add_argument('--margin_same', type=float, default=0.5, help='loss margin: same instance')
parser.add_argument('--margin_diff', type=float, default=1., help='loss margin: different instance')
parser.add_argument('--use_vdm', type=bool, default=True, help='use the valid distance matrix for loss')
parser.add_argument('--use_asm', type=bool, default=True, help='use the attention score matrix for loss')

parser.add_argument('--verbose', action='store_true', help='if specified, use depthconv')
# parser.add_argument('--input_list', type=str, default='t', help='test data list')
parser.add_argument('--restore_dir', type=str, default='checkpoint/', help='Directory that stores all training logs and trained models')
parser.add_argument('--point_dim', type=int, default=6, help='dim of point cloud,XYZ,NxNyNz or RGB')
parser.add_argument('--center_socre_th', type=float, default=0.6, help='valid center score: 0.4~0.8')
parser.add_argument('--backbone', type=str, default='dgcnn', help='backbone: pointnet,dgcnn,Xnet')
parser.add_argument('--r_nms', type=float, default=.1, help='bunny:, A: .5~.6 / B: 0.8~1 / C:.4~.5 / ring: 0.1 / gear: .08')
parser.add_argument('--input_list', type=str, default='datas/ring_train.txt', help='Input data list file')
args = parser.parse_args()
