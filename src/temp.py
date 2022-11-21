from arguments import get_args
from model.CmmlLayer import CmmlModel


def main():
    
    args = get_args()
    model = CmmlModel(args)

    model = model.cuda()
    
    return 0

if __name__ == '__main__':
    main()
    

