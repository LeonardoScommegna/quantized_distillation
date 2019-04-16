from os import path
import sys

print(path.abspath('../bcfind'))
sys.path.append(path.abspath('../bcfind'))
from train_deconvolver.models.FC_teacher_max_p import FC_teacher_max_p
from train_deconvolver.models.FC_student import FC_student
from train_deconvolver.data_reader import *
import os
import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader
import cnn_models.conv_forward_model as convForwModel
from cnn_models.help_fun import bcfind_evaluateModel, bcfind_forward_and_backward


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

    parser.add_argument('models_save_path', metavar='models_save_path', type=str,
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

    parser.add_argument('-k_teacher', '--kernel_size_teacher', dest='kernel_size_teacher', type=int,
                        default=3,
                        help="""size of the cubic kernel of the teacher, provide only an integer e.
                        g. kernel_size 3""")

    parser.add_argument('-f_teacher', '--initial_filters_teacher', dest='initial_filters_teacher',
                        type=int,
                        default=4,
                        help="""Number of filters in the initial conv layer for the teacher""")

    parser.add_argument('-k_student', '--kernel_size_student', dest='kernel_size_student', type=int,
                        default=3,
                        help="""size of the cubic kernel of the student, provide only an integer e.
                            g. kernel_size 3""")

    parser.add_argument('-f_student', '--initial_filters_student', dest='initial_filters_student',
                        type=int,
                        default=4,
                        help="""Number of filters in the initial conv layer for the student""")

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
    parser.add_argument('-first', dest='first_time', action ='store_true',
                        help="pass this argument the first you train a model")

    parser.add_argument('--model_manager_save_file', dest = 'model_manger_save_file', type =str,
                        default=None)





    parser.set_defaults(first_time =False)
    return parser




def main():
    parser = get_parser()
    args = parser.parse_args()
    USE_CUDA = torch.cuda.is_available()

    try:
        os.mkdir(args.models_save_path)
    except:
        pass
    save_file_manger = 'model_manager_bcfind.tst' if not args.model_manger_save_file else args.model_manger_save_file
    bcfind_manager = model_manager.ModelManager(save_file_manger, 'model_manager', create_new_model_manager=args.first_time )

    #non esiste
    save_path = os.path.join(args.models_save_path, args.name_dir)

    for x in bcfind_manager.list_models():
        if bcfind_manager.get_num_training_runs(x) >= 1:
            s = '{}; Last prediction acc: {}, Best prediction acc: {}'.format(x,
                                                                              bcfind_manager.load_metadata(x)[1][
                                                                                  'predictionAccuracy'][-1],
                                                                              max(bcfind_manager.load_metadata(x)[1][
                                                                                      'predictionAccuracy']))
            print(s)

    try:
        os.mkdir(save_path)
    except:
        pass

    '''Train Data Loader'''
    train_dataframe = pd.read_csv(args.train_csv_path, comment="#",
                                  index_col=0, dtype={"img_name": str})
    patch_size = int(str(pd.read_csv(args.train_csv_path, nrows=1, header=None).
                         take([0])).split("=")[-1])  # workaround
    train_dataset = DataReaderWeight(args.img_dir, args.gt_dir,
                                     args.weight_dir,
                                     train_dataframe, patch_size)
    train_loader = DataLoader(train_dataset, args.batch_size,
                              shuffle=True, num_workers=args.n_workers)

    '''Validation/Test Data Loader'''

    val_test_dataframe = pd.read_csv(args.val_test_csv_path, comment="#",
                                     index_col=0, dtype={"name": str})
    test_dataframe = val_test_dataframe[(val_test_dataframe['split'] == 'VAL')]

    test_dataset = DataReaderSubstackTest(args.img_dir, args.gt_dir, args.marker_dir, test_dataframe)

    test_loader = DataLoader(test_dataset, 1, shuffle=False, num_workers=args.n_workers)

    ''' Teacher and Student networks'''

    teacher= FC_teacher_max_p(args.initial_filters_teacher, k_conv=args.kernel_size_teacher).cuda()

    teacher.load_state_dict(torch.load(args.teacher_path, map_location=args.device))

    student = FC_student(args.initial_filters_student, k_conv=args.kernel_size_student).cuda()

    student_model_name = args.name_dir+'{}bit'.format(args.n_bit)

    student_model_path = os.path.join(args.models_save_path, student_model_name)
    distillationOptions = {}
    if not student_model_name in bcfind_manager.saved_models:
        bcfind_manager.add_new_model(student_model_name, student_model_path,arguments_creator_function=distillationOptions)


    bcfind_manager.train_model(student, model_name=student_model_name,
                               train_function=convForwModel.train_model,
                               arguments_train_function={'epochs_to_train': args.epochs,
                                                         'use_distillation_loss': True,
                                                         'teacher_model': teacher,
                                                         'quantizeWeights': True,
                                                         'numBits': args.n_bit,
                                                         'bucket_size': 256,
                                                         'quantize_first_and_last_layer': False,
                                                         'loss_function': bcfind_forward_and_backward,
                                                         'eval_function': bcfind_evaluateModel,
                                                         'soma_weight': args.soma_weight
                                                         },
                               train_loader=train_loader, test_loader=test_loader)



if __name__=="__main__":
    main()

    pass