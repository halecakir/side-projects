"""TODO"""
from argparse import ArgumentParser
from models.sentence_classification import run


def parse_arguments():
    """TODO"""
    parser = ArgumentParser()
    parser.add_argument(
        "--permission-type",
        dest="permission_type",
        help="Test permission",
        default="N/A",
    )
    parser.add_argument(
        "--saved-parameters-dir",
        dest="saved_parameters_dir",
        help="Saved model parameters directory",
        metavar="FILE",
        default="N/A",
    )
    parser.add_argument(
        "--saved-data",
        dest="saved_data",
        help="Saved embeddings, train/test & vocab data",
        metavar="FILE",
        default="N/A",
    )
    parser.add_argument(
        "--saved-reviews",
        dest="saved_reviews",
        help="Saved review data",
        metavar="FILE",
        default="N/A",
    )
    parser.add_argument(
        "--saved-predicted-reviews",
        dest="saved_predicted_reviews",
        help="Saved rpredicted eview data",
        metavar="FILE",
        default="N/A",
    )
    parser.add_argument(
        "--model-checkpoint",
        dest="model_checkpoint",
        help="Saved model file",
        metavar="FILE",
        default="N/A",
    )
    parser.add_argument(
        "--outdir",
        dest="outdir",
        help="Output directory",
        metavar="FILE",
        default="N/A",
    )
    parser.add_argument(
        "--stemmer", dest="stemmer", help="Apply word stemmer", default="porter"
    )
    args = parser.parse_args()
    return args


def main():
    """TODO"""
    args = parse_arguments()
    run(args)


if __name__ == "__main__":
    main()
