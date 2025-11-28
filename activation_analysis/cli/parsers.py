import argparse
import configs

class MainParser:
    def parse_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Perform different analysis on transformers models.",
            add_help=True,
        )
        parser.add_argument(
            "-m",
            "--model",
            type=str,
            default=configs.VIT_BASE_PATCH16_224,
            help="Model name.",
            choices=(configs.VIT_CLASSIFICATION_CONFIGS + configs.TEXT_MODELS),
        )
        parser.add_argument(
            "-D",
            "--dataset",
            type=str,
            default=configs.IMAGENET,
            help="Dataset name.",
            choices=[configs.IMAGENET, configs.COCO, configs.GLUE_SST2, configs.GLUE_MNLI],
        )
        parser.add_argument(
            "-b",
            "--batch-size",
            type=int,
            default=configs.DEFAULT_BATCH_SIZE,
            help="Batch size.",
        )
        parser.add_argument(
            "-p",
            "--precision",
            type=str,
            default=configs.FP32,
            help="Precision of the model and inputs.",
            choices=[configs.FP16, configs.FP32],
        )
        parser.add_argument(
            "-d",
            "--device",
            type=str,
            default=configs.GPU_DEVICE,
            help="Device to run the model.",
            choices=[
                configs.CPU,
                configs.GPU_DEVICE,
                configs.GPU_DEVICE1,
                configs.GPU_DEVICE2,
                configs.GPU_DEVICE3,
            ],
        )
        parser.add_argument(
            "-n", "--num-samples", type=int, default=-1, help="Number of samples to use."
        )
        parser.add_argument(
            "-s", "--seed", type=int, default=configs.SEED, help="Random seed."
        )
        parser.add_argument(
            "-S",
            "--shuffle-dataset",
            default=False,
            action="store_true",
            help="Shuffle the dataset or not.",
        )
        parser.add_argument(
            "-v", "--verbose", action="store_true", help="Verbose mode.", default=False
        )
        parser.add_argument(
            "--run-low-conf",
            action="store_true",
            help="Run the model on low confidence samples only.",
            default=False,
        )

        return parser.parse_args()


class GPT2TrainParser:
    def parse_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Train GPT-2 for sentiment classification.",
            add_help=True,
        )
        parser.add_argument(
            "-m",
            "--model",
            type=str,
            default=configs.GPT2,
            help="Model name.",
            choices=configs.TEXT_MODELS,
        )
        parser.add_argument(
            "-D",
            "--dataset",
            type=str,
            default=configs.IMAGENET,
            help="Dataset name.",
            choices=[configs.GLUE_SST2, configs.GLUE_MNLI],
        )
        parser.add_argument(
            "-b",
            "--batch-size",
            type=int,
            default=configs.DEFAULT_BATCH_SIZE,
            help="Batch size.",
        )
        parser.add_argument(
            "-p",
            "--precision",
            type=str,
            default=configs.FP32,
            help="Precision of the model and inputs.",
            choices=[configs.FP16, configs.FP32],
        )
        parser.add_argument(
            "-d",
            "--device",
            type=str,
            default=configs.GPU_DEVICE,
            help="Device to run the model.",
            choices=[
                configs.CPU,
                configs.GPU_DEVICE,
                configs.GPU_DEVICE1,
                configs.GPU_DEVICE2,
                configs.GPU_DEVICE3,
            ],
        )
        parser.add_argument(
            "-s", "--seed", type=int, default=configs.SEED, help="Random seed."
        )
        parser.add_argument(
            "-S",
            "--shuffle-dataset",
            default=False,
            action="store_true",
            help="Shuffle the dataset or not.",
        )
        parser.add_argument(
            "-v", "--verbose", action="store_true", help="Verbose mode.", default=False
        )
        parser.add_argument(
            "-e",
            "--num-epochs",
            type=int,
            default=3,
            help="Number of training epochs.",
        )
        parser.add_argument(
            "-l",
            "--lr",
            type=float,
            default=5e-5,
            help="Learning rate.",
        )
        parser.add_argument(
            "-o",
            "--output-dir",
            type=str,
            default="./gpt2-sst2-finetuned",
            help="Output directory to save the model.",
        )

        return parser.parse_args()
    

class ActivationAnalysisPlotParser:
    def parse_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Activation Analysis Plot parameters.",
            add_help=True,
        )
        parser.add_argument(
            "-v", "--verbose", action="store_true", help="Verbose mode.", default=False
        )
        parser.add_argument(
            "-i",
            "--input-dir",
            type=str,
            default="data/profiling/",
            help="Input directory to load activations from.",
        )
        parser.add_argument(
            "-o",
            "--output-dir",
            type=str,
            default="data/plots/",
            help="Output directory to save activations.",
        )

        return parser.parse_args()