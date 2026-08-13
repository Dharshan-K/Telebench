import argparse
import os
import sys
from importlib.resources import files

from ..benchmark import Benchmark

DESCRIPTION = "Run the speech degradation, noise-mixing and ASR-evaluation benchmark pipeline."


def warn(message):
    print(f"WARNING: {message}")


def build_parser():
    parser = argparse.ArgumentParser(prog="telebench", description=DESCRIPTION)
    parser.add_argument(
        "-c", "--config",
        default="config.yaml",
        help="Path to the config yaml file (default: config.yaml).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "degrade",
        help="Run fft -> downsample -> mu-law degradation on preprocess.input_dir.",
    )

    subparsers.add_parser(
        "noise",
        help="Add noise to the degraded speech files (noise_mixing stage).",
    )

    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Transcribe original vs degraded audio via OpenAI-compatible APIs and report metrics.",
    )
    eval_parser.add_argument(
        "-e", "--endpoint",
        action="append",
        metavar="NAME",
        help="Evaluate only the endpoint(s) matching this name (repeatable). "
             "Defaults to all endpoints in config.",
    )

    subparsers.add_parser(
        "endpoints",
        help="List the configured evaluation endpoints.",
    )

    subparsers.add_parser(
        "eval",
        help="Run degrade, noise, then evaluate.",
    )

    subparsers.add_parser(
        "info",
        help="Load and validate the config, then show its paths and endpoints.",
    )

    subparsers.add_parser(
        "default",
        help="Run the full pipeline (degrade -> noise -> evaluate) using the config and audio bundled in the package.",
    )

    return parser


def _print_config(benchmark):
    preprocess = benchmark.config["preprocess"]
    noise_mixing = benchmark.config["noise_mixing"]
    evaluation = benchmark.config["evaluation"]
    print(f"preprocess input:  {preprocess['input_dir']}")
    print(f"preprocess output: {preprocess['output_dir']}")
    print(f"noise input:       {noise_mixing['input_dir']}")
    print(f"noise dir:         {noise_mixing['noise_dir']}")
    print(f"noise output:      {noise_mixing['output_dir']}")
    print(f"eval original:     {evaluation['original_dir']}")
    print(f"eval degraded:     {evaluation['degraded_dir']}")
    print(f"eval output:       {evaluation.get('output_path', '(none)')}")
    _print_endpoints(evaluation)


def _print_endpoints(evaluation):
    print("\nendpoints:")
    for endpoint in evaluation["endpoints"]:
        name = endpoint.get("name") or f"{endpoint['base_url']}:{endpoint['model']}"
        print(f"  {name}")
        print(f"    base_url: {endpoint['base_url']}")
        print(f"    model:    {endpoint['model']}")
        print(f"    api_key_env: {endpoint.get('api_key_env', '(OPENAI_API_KEY)')}")


def _select_endpoints(benchmark, requested):
    endpoints = benchmark.config["evaluation"]["endpoints"]
    if not requested:
        return endpoints
    names = set(requested)
    selected = [
        ep for ep in endpoints
        if (ep.get("name") or f"{ep['base_url']}:{ep['model']}") in names
    ]
    found = {
        ep.get("name") or f"{ep['base_url']}:{ep['model']}" for ep in selected
    }
    missing = sorted(names - found)
    if missing:
        warn(f"Unknown endpoint(s): {missing}, skipping.")
    return selected


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "default":
            config_path = files("telebench").joinpath("config.yaml")
            os.chdir(os.path.dirname(str(config_path)))
            benchmark = Benchmark("config.yaml")
        else:
            benchmark = Benchmark(args.config)

        if args.command == "info":
            _print_config(benchmark)
            return 0

        if args.command == "endpoints":
            _print_endpoints(benchmark.config["evaluation"])
            return 0

        if args.command == "evaluate":
            selected = _select_endpoints(benchmark, args.endpoint)
            benchmark.config["evaluation"]["endpoints"] = selected
            if not selected:
                warn("no endpoints to evaluate.")
                return 0

        if args.command in ("eval", "default"):
            print("Running degrade...")
            benchmark.process("degrade")
            print("Running noise mixing...")
            benchmark.process("noise")
            print("Running evaluation...")
            benchmark.process("evaluate")
        else:
            print(f"Running {args.command}...")
            benchmark.process(args.command)

        print("Done.")
        return 0
    except Exception as exc:
        warn(str(exc))
        return 0


if __name__ == "__main__":
    sys.exit(main())
