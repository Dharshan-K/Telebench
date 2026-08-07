import os
import random

import soundfile as sf
import yaml

from core.evaluate import EvaluationRunner, VALID_METRICS
from core.process import Process
from dataloader.localloader import LocalLoader

REQUIRED_SECTIONS = {"dataset", "preprocess", "noise_mixing", "evaluation"}


class Benchmark:
    def __init__(self, config_path="./example.yml"):
        self.config = self.parse_yaml(config_path)
        self.validate(self.config)
        self._create_output_dirs()
        self.loaders = {}
        self.data = None

    @staticmethod
    def parse_yaml(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    @staticmethod
    def validate(config):
        missing = REQUIRED_SECTIONS - set(config)
        if missing:
            raise ValueError(f"Config missing required sections: {missing}")

        def require_keys(section_name, section, keys):
            missing_keys = set(keys) - set(section)
            if missing_keys:
                raise ValueError(
                    f"Config section '{section_name}' missing required keys: "
                    f"{missing_keys}"
                )

        preprocess = config["preprocess"]
        require_keys(
            "preprocess",
            preprocess,
            {"input_dir", "output_dir"},
        )
        if not os.path.isdir(preprocess["input_dir"]):
            raise ValueError(
                f"preprocess.input_dir does not exist: {preprocess['input_dir']}"
            )

        noise_mixing = config["noise_mixing"]
        require_keys(
            "noise_mixing",
            noise_mixing,
            {"snr_db_min", "snr_db_max", "input_dir", "noise_dir", "output_dir"},
        )
        if noise_mixing["snr_db_min"] >= noise_mixing["snr_db_max"]:
            raise ValueError("noise_mixing.snr_db_min must be < snr_db_max")
        if not os.path.isdir(noise_mixing["noise_dir"]):
            raise ValueError(
                f"noise_mixing.noise_dir does not exist: {noise_mixing['noise_dir']}"
            )
        noise_loader = LocalLoader(noise_mixing["noise_dir"])
        noise_files = [f for f in noise_loader]
        if not noise_files:
            raise ValueError(
                f"No audio files found in noise_mixing.noise_dir: {noise_mixing['noise_dir']}"
            )

        evaluation = config["evaluation"]
        require_keys(
            "evaluation",
            evaluation,
            {"endpoints", "original_dir", "degraded_dir"},
        )
        for endpoint in evaluation["endpoints"]:
            if "base_url" not in endpoint or "model" not in endpoint:
                raise ValueError(
                    "Each evaluation.endpoints entry must have 'base_url' and 'model'"
                )
        metrics = evaluation.get("metrics", ["wer", "cer"])
        unknown = set(metrics) - VALID_METRICS
        if unknown:
            raise ValueError(
                f"evaluation.metrics contains unknown metrics: {unknown}. "
                f"Valid: {sorted(VALID_METRICS)}"
            )

    def _create_output_dirs(self):
        for directory in (
            self.config["dataset"]["speech"]["output_dir"],
            self.config["dataset"]["noise"]["output_dir"],
            self.config["preprocess"]["output_dir"],
            self.config["noise_mixing"]["output_dir"],
        ):
            os.makedirs(directory, exist_ok=True)

    def _loader_for(self, directory):
        if directory not in self.loaders:
            self.loaders[directory] = LocalLoader(directory)
        return self.loaders[directory]

    def load(self, directory=None):
        if directory is None:
            directory = self.config["preprocess"]["input_dir"]
        self.data = [file for file in self._loader_for(directory)]
        return self.data

    def process(self, mode="degrade"):
        if mode == "degrade":
            self._process_degrade()
        elif mode == "noise":
            self._process_noise()
        elif mode == "evaluate":
            self._process_evaluate()
        else:
            raise ValueError(
                f"Unknown mode: {mode!r}. Use 'degrade', 'noise' or 'evaluate'."
            )

    def _process_evaluate(self):
        evaluation = self.config["evaluation"]
        for key in ("original_dir", "degraded_dir"):
            if not os.path.isdir(evaluation[key]):
                raise ValueError(
                    f"evaluation.{key} does not exist: {evaluation[key]}. "
                    "Run 'degrade' and 'noise' steps first."
                )
        runner = EvaluationRunner(evaluation)
        runner.run()

    def _process_degrade(self):
        preprocess = self.config["preprocess"]
        output_dir = preprocess["output_dir"]
        for file in self.load():
            proc = Process(
                str(file),
                normalize_scale=preprocess.get("normalize_scale", 32768),
            )
            proc.process()
            proc.fourier_transform()
            proc.mu_law()
            out_path = os.path.join(output_dir, file.name)
            sf.write(out_path, proc.get_data(), proc.sample_rate)

    def _process_noise(self):
        noise_mixing = self.config["noise_mixing"]
        input_dir = noise_mixing["input_dir"]
        noise_dir = noise_mixing["noise_dir"]
        output_dir = noise_mixing["output_dir"]
        noise_files = [str(f) for f in self._loader_for(noise_dir)]
        if not noise_files:
            raise ValueError(
                f"No audio files found in noise_dir: {noise_dir}"
            )
        if not os.path.isdir(input_dir):
            raise ValueError(
                f"noise_mixing.input_dir does not exist: {input_dir}. "
                "Run the 'degrade' step first."
            )
        input_files = [f for f in self._loader_for(input_dir)]
        if not input_files:
            raise ValueError(
                f"No audio files found in noise_mixing.input_dir: {input_dir}. "
                "Run the 'degrade' step first."
            )
        for file in self.load(input_dir):
            noise_file = random.choice(noise_files)
            out_path = os.path.join(output_dir, file.name)
            proc = Process(
                str(file),
                noise_file=noise_file,
                snr_db_min=noise_mixing["snr_db_min"],
                snr_db_max=noise_mixing["snr_db_max"],
            )
            proc.process()
            proc.add_noise(output_path=out_path)
