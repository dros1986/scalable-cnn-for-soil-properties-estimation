import argparse

def parse_train_arguments():
    # get arguments
    parser = argparse.ArgumentParser()
    # hyperparametrs
    parser.add_argument("-bs", "--batchsize", help="Batch size.",
    					default=100, type=int)
    parser.add_argument("-ne", "--nepochs", help="Number of epochs.",
    					default=500, type=int)
    parser.add_argument("-lr", "--learning_rate", help="Learning rate.",
    					default=1e-3, type=float)
    parser.add_argument("-wd", "--weight_decay", help="Weight decay.",
    					default=0, type=float)
    parser.add_argument("-sf", "--start_from", help="Starts from epoch.",
    					default=-1, type=int)
    # loss and quantization
    parser.add_argument("-loss", "--loss", help="Loss function.",
    					default='l1', type=str)
    parser.add_argument("-nbins", "--nbins", help="Number of bins for binarization.",
    					default=10, type=int)
    # validation
    parser.add_argument("-val", "--val", help="Validation function.",
    					default='r2', type=str,  choices=['mae', 'mse', 'rmse', 'r2'])
    # network
    parser.add_argument("-powf", "--powf", help="Power of 2 of filters.",
    					default=3, type=int)
    parser.add_argument("-max_powf", "--max_powf", help="Max power of 2 of filters.",
    					default=8, type=int)
    parser.add_argument("-insz", "--insz", help="Input size.",
    					default=1024, type=int)
    parser.add_argument("-minsz", "--minsz", help="Min size.",
    					default=8, type=int)
    parser.add_argument("-nbsr", "--nbsr", help="Number of blocks same resolution.",
    					default=1, type=int)
    parser.add_argument("-leak", "--leak", help="Leak of relu. If 0, normal relu.",
    					default=0, type=float)
    parser.add_argument("-mom", "--momentum", help="Batchnorm momentum.",
    					default=0.01, type=float)
    # normalization and quantization
    parser.add_argument("-norm", "--norm_type", help="Normalization used in nir/swir.",
                        default='std_instance', type=str)
    # data
    parser.add_argument("-dev", "--device", help="Device.",
    					default='cpu', type=str)
    parser.add_argument("-exp", "--experiment", help="Name of experiment.",
    					default='experiment1', type=str)
    parser.add_argument("-tcsv", "--train_csv", help="Lucas train csv file.",
    					default='/home/flavio/datasets/LucasLibrary/LucasTopsoil/LUCAS.SOIL_corr_FULL_train.csv')
    parser.add_argument("-vcsv", "--val_csv", help="Lucas train csv file.",
    					default='/home/flavio/datasets/LucasLibrary/LucasTopsoil/LUCAS.SOIL_corr_FULL_val.csv')
    parser.add_argument("-srcp", "--src_prefix", help="Prefix of input signal as specified in the csv", \
                        default="spc.")
    parser.add_argument("-tvars", "--tgt_vars", nargs="+", \
                    default=['coarse','clay','silt','sand','pH.in.CaCl2','pH.in.H2O','OC','CaCO3','N','P','K','CEC'], \
                    help="Name of the target variables as specified in the csv file.")
    args = parser.parse_args()
    return args



def parse_test_arguments():
    # get arguments
    parser = argparse.ArgumentParser()
    # hyperparametrs
    parser.add_argument("-bs", "--batchsize", help="Batch size.",
    					default=100, type=int)
    # loss and quantization
    parser.add_argument("-nbins", "--nbins", help="Number of bins for binarization.",
    					default=10, type=int)
    # network
    parser.add_argument("-powf", "--powf", help="Power of 2 of filters.",
    					default=3, type=int)
    parser.add_argument("-max_powf", "--max_powf", help="Max power of 2 of filters.",
    					default=8, type=int)
    parser.add_argument("-insz", "--insz", help="Input size.",
    					default=1024, type=int)
    parser.add_argument("-minsz", "--minsz", help="Min size.",
    					default=8, type=int)
    parser.add_argument("-nbsr", "--nbsr", help="Number of blocks same resolution.",
    					default=1, type=int)
    parser.add_argument("-leak", "--leak", help="Leak of relu. If 0, normal relu.",
    					default=0, type=float)
    parser.add_argument("-mom", "--momentum", help="Batchnorm momentum.",
    					default=0.01, type=float)
    # normalization and quantization
    parser.add_argument("-norm", "--norm_type", help="Normalization used in nir/swir.",
                        default='std_instance', type=str)
    # data
    parser.add_argument("-dev", "--device", help="Device.",
    					default='cpu', type=str)
    parser.add_argument("-exp", "--experiment", help="Name of experiment.",
    					default='experiment1', type=str)
    parser.add_argument("-tcsv", "--test_csv", help="Lucas train csv file.",
    					default='/home/flavio/datasets/LucasLibrary/LucasTopsoil/LUCAS.SOIL_corr_FULL_val.csv')
    parser.add_argument("-srcp", "--src_prefix", help="Prefix of input signal as specified in the csv", \
                        default="spc.")
    parser.add_argument("-tvars", "--tgt_vars", nargs="+", \
                    default=['coarse','clay','silt','sand','pH.in.CaCl2','pH.in.H2O','OC','CaCO3','N','P','K','CEC'], \
                    help="Name of the target variables as specified in the csv file.")
    args = parser.parse_args()
    return args
