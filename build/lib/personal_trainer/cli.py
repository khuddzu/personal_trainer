import argparse
import os

def main():
    parser = argparse.ArgumentParser(prog='cli', 
            description = 'Command Line Arguments for personal_trainer package.')
    subparsers = parser.add_subparsers(dest='workflow', help='select workflow', required=True)
    common_parser = argparse.ArgumentParser(description="Common parser", add_help=False)
    common_parser.add_argument('-v', '--verbose', action='store_true')
    parser_init = subparsers.add_parser('init', help='Initiate a project', parents=[common_parser])
    args = parser.parse_args()

    if args.workflow == "init":
            project_name = Path('training_center')
            assert not project_name.exists(
            ), (f"{project_name.absolute().as_posix()} already exist, please choose another project name (using flag -p newname) "
                "or remove the old directory")
            assert '/' not in project_name.as_posix(), 'project name should not contain /'
            print(f"initiating personal_trainer at {project_name.absolute().as_posix()} ")
            os.makedirs(project_name.as_posix())
            os.makedirs((project_name / 'logs').as_posix())
            os.makedirs((project_name / 'models').as_posix())


if __name__ == '__main__':
    main()
                                                                                                                       
