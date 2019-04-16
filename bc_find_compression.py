from os import path
import sys
sys.path.append(path.abspath('../bcfind'))
from bcfind.train_deconvolver.models import FC_teacher_max_p
from bcfind.train_deconvolver.models import FC_student

import os
import argparse

import model_manager

def get_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.
                                     ArgumentDefaultsHelpFormatter)
    parser.add_argument('teacher_path', metavar='teacher_path', type=str,
                        help="""Path to the teacher pth model""")
    parser.add_argument('n_bit', metavar='n_bit', type = int,
                        help =""" Number of bit in the comrpessed model """)

    parser.add_argument('train_csv_path', metavar='train_csv_path', type=str,
                        help="""path to csv patches """)

    parser.add_argument('val_test_csv_path', metavar='val_test_csv_path', type=str,
                        help="""path to val/train csv""")

    parser.add_argument('img_dir', metavar='img_dir', type=str,
                        help="""Directory contaning the collection
                        of pth images""")

    parser.add_argument('gt_dir', metavar='gt_dir', type=str,
                        help="""Directory contaning the collection
                        of pth gt images""")

    parser.add_argument('weight_dir', metavar='weight_dir', type=str,
                        help="""Directory contaning the collection
                        of weighted_map""")

    parser.add_argument('marker_dir', metavar='marker_dir', type=str,
                        help="""Directory contaning the collection
                            of gt markers""")

    parser.add_argument('model_save_path', metavar='model_save_path', type=str,
                        help="""directory where the models will be saved""")

    parser.add_argument('device', metavar='device', type=str,
                        help="""device to use during training
                        and validation phase, e.g. cuda:0""")

    parser.add_argument('-w', '--weight', dest='soma_weight', type=float,
                        default=1.0,
                        help=""" wheight for class soma cell""")

    parser.add_argument('-b', '--batch_size', dest='batch_size', type=int,
                        default=8,
                        help="""batch size, integer number""")

    parser.add_argument('-t', '--n_workers', dest='n_workers', type=int,
                        default=4,
                        help="""number of workers for data loader""")

    parser.add_argument('--epochs', dest='epochs', type=int,
                        default=100,
                        help="""number of training epochs""")

    parser.add_argument('-k', '--kernel_size', dest='kernel_size', type=int,
                        default=3,
                        help="""size of the cubic kernel, provide only an integer e.
                        g. kernel_size 3""")

    parser.add_argument('-f', '--initial_filters', dest='initial_filters',
                        type=int,
                        default=4,
                        help="""Number of filters in the initial conv layer""")

    parser.add_argument('--lr', dest='lr', type=float,
                        default=0.001,
                        help="""learning rate""")

    parser.add_argument('-l', '--root_log_dir', dest='root_log_dir', type=str,
                        default=None,
                        help="""log directory for tensorbard""")

    parser.add_argument('-n', '--name_dir', dest='name_dir', type=str,
                        default=None,
                        help="""name of the directory where  model will
                        be stored""")
    return parser


def additional_namespace_arguments(parser):
    '''
    parser.add_argument('-r', '--hi_local_max_radius', metavar='r', dest='hi_local_max_radius',
                        action='store', type=float, default=6,
                        help='Radius of the seed selection ball (r)')

    parser.add_argument('-fl', '--floating_point', dest='floating_point', action='store_true',
                        help='If true, cell centers are saved in floating point.')

    parser.add_argument('-m', '--min_second_threshold', metavar='min_second_threshold', dest='min_second_threshold',
                        action='store', type=int, default=15,
                        help="""If the foreground (second threshold in multi-Kapur) is below this value
                                   then the substack is too dark and assumed to contain no soma""")
    parser.add_argument('-loc', '--local', dest='local', action='store_true',
                        help='Perform local processing by dividing the volume in 8 parts.')
    # parser.set_defaults(local=True)
    parser.add_argument('-ts', '--seeds_filtering_mode', dest='seeds_filtering_mode',
                        action='store', type=str, default='soft',
                        help="Type of seed selection ball ('hard' or 'soft')")
    parser.add_argument('-R', '--mean_shift_bandwidth', metavar='R', dest='mean_shift_bandwidth',
                        action='store', type=float, default=5.5,
                        help='Radius of the mean shift kernel (R)')
    parser.add_argument('-s', '--save_image', dest='save_image', action='store_true',
                        help='Save debugging substack for visual inspection (voxels above threshold and colorized clusters).')
    parser.add_argument('-M', '--max_expected_cells', metavar='max_expected_cells', dest='max_expected_cells',
                        action='store', type=int, default=10000,
                        help="""Max number of cells that may appear in a substack""")
    parser.add_argument('-p', '--pair_id', dest='pair_id',
                        action='store', type=str,
                        help="id of the pair of views, e.g 000_090. A folder with this name will be created inside outdir/substack_id")
    parser.add_argument('-D', '--max_cell_diameter', dest='max_cell_diameter', type=float, default=16.0,
                        help='Maximum diameter of a cell')
    parser.add_argument('-d', '--manifold-distance', dest='manifold_distance', type=float, default=None,
                        help='Maximum distance from estimated manifold to be included as a prediction')
    parser.add_argument('-c', '--curve', dest='curve', action='store_true', help='Make a recall-precision curve.')
    parser.add_argument('-g', '--ground_truth_folder', dest='ground_truth_folder', type=str, default=None,
                        help='folder containing merged marker files (for multiview images)')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Verbose output.')
    parser.add_argument('--do_icp', dest='do_icp', action='store_true',
                        help='Use the ICP matching procedure to evaluate the performance')

    parser.add_argument('-e', dest='evaluation', action='store_true')
    '''
    parser.set_defaults(hi_local_max_radius=6)
    #parser.set_defaults()
    parser.set_defaults(min_second_threshold =15)
    parser.set_defaults(mean_shift_bandwidth = 5.5)
    parser.set_defaults(seeds_filtering_mode='soft')
    parser.set_defaults(max_expected_cells = 10000)
    parser.set_defaults(max_cell_diameter = 16.0)
    parser.set_defaults(verbose = False)
    parser.set_defaults(save_image=False)
    parser.set_defaults(evaluation=True)
    parser.set_defaults(do_icp= True)
    parser.set_defaults(manifold_distance = 40)


def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def criterium_to_save():
    return True

def main():
    parser = get_parser()
    additional_namespace_arguments(parser)
    args = parser.parse_args()



if __name__=="__main__":
    main()

    pass