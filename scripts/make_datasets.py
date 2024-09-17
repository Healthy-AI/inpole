import argparse

from inpole.data.preprocessing import make_adni, make_sepsis, make_copd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--adni_path', type=str)
    parser.add_argument('--sepsis_path', type=str)
    parser.add_argument('--copd_path', type=str)
    parser.add_argument('--out_path', type=str)
    args = parser.parse_args()

    if args.adni_path:
        make_adni(args.adni_path, args.out_path)
    
    if args.sepsis_path:
        make_sepsis(args.sepsis_path, args.out_path)

    if args.copd_path:
        make_copd(args.copd_path, args.out_path)
