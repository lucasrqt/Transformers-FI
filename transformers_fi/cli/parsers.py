import argparse
import configs
import statistical_fi

class MainParser:
    def parse_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Perform high-level fault injections on ViT model according neutron beam fault model.",
            add_help=True,
        )
        parser.add_argument(
            "-m",
            "--model",
            type=str,
            default=configs.VIT_BASE_PATCH16_224,
            help="Model name.",
            choices=configs.LATS_MODELS,
        )
        parser.add_argument(
            "-D",
            "--dataset",
            type=str,
            default=configs.IMAGENET,
            help="Dataset name.",
            choices=configs.LATS_DATASETS
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
            choices=[configs.FP32],
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
            "-M",
            "--microop",
            type=str,
            default=None,
            help="Microoperation to inject the fault.",
            choices=configs.MICROBENCHMARK_MODULES,
        )
        parser.add_argument(
            "--target-layer",
            type=int,
            default=-1,
            help="Target layer for the fault injection.",
        )
        parser.add_argument(
            "--injection-type",
            type=lambda it: statistical_fi.InjectionType[it],
            default=statistical_fi.InjectionType.RANDOM,
            help="Type of injection to perform.",
            choices=list(statistical_fi.InjectionType),
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
            "--fault-model-threshold",
            type=float,
            default=1e-04,
            help="Threshold for the fault model data.",
        )
        parser.add_argument(
            "--inject-on-correct-predictions",
            action="store_true",
            help="Inject faults only on correct predictions.",
            default=False,
        )
        parser.add_argument(
            "--load-critical",
            action="store_true",
            help="Only load the images that are critical for the fault injection.",
            default=False,
        )
        parser.add_argument(
            "--save-critical-logits",
            action="store_true",
            help="Save the logits of the critical images.",
            default=False,
        )
        parser.add_argument(
            "--save-top5prob",
            action="store_true",
            help="Save the top 5 probabilities of the critical images.",
            default=False,
        )
        parser.add_argument(
            "-v", "--verbose", action="store_true", help="Verbose mode.", default=False
        )
        parser.add_argument(
            "--bitflip-position",
            type=int,
            default=0,
            help="Specific bit position to flip 0 (for LSB) to 31. If None, a random position will be used.",
            choices=configs.BITFLIP_POSITIONS,
        )
        parser.add_argument(
            "-n", "--nsamples", type=int, default=-1, help="Number of samples to use."
        )
        parser.add_argument(
            "--range-restriction-mode",
            type=lambda rrm: statistical_fi.RangeRestrictionMode[rrm],
            default=statistical_fi.RangeRestrictionMode.NONE,
            help="Mode for range restriction after fault injection.",
            choices=list(statistical_fi.RangeRestrictionMode),
        )

        return parser.parse_args()


class ResultsParser:
    def parse_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Parse and display fault injection results.",
            add_help=True,
        )
        parser.add_argument(
            "-r",
            "--results-dir",
            type=str,
            required=True,
            help="Directory containing the fault injection results.",
        )
        parser.add_argument(
            "-o",
            "--output-dir",
            type=str,
            default="./parsed_results",
            help="Directory to save the parsed results.",
        )
        parser.add_argument(
            "-e", 
            "--erase-output-dir", 
            action="store_true", 
            help="Erase the output directory if it exists.",
            default=False
        )
        parser.add_argument(
            "-v", "--verbose", action="store_true", help="Verbose mode.", default=False
        )

        return parser.parse_args()