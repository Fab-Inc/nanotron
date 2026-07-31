import argparse
import os
import subprocess
from pathlib import Path
from shutil import rmtree
from typing import Optional

from dotenv import load_dotenv
from typer import Option
from typing_extensions import Annotated

from nanotron.config import ParallelismArgs, get_config_from_file
from nanotron.config.lighteval_config import (
    GenerationArgs,
    LightEvalConfig,
    LightEvalLoggingArgs,
    LightEvalTasksArgs,
)

load_dotenv(override=True)


CACHE_DIR: str = os.getenv("HF_HOME", "/scratch")

HELP_PANNEL_NAME_1 = "Common Paramaters"
HELP_PANNEL_NAME_2 = "Logging Parameters"
HELP_PANNEL_NAME_3 = "Debug Paramaters"
HELP_PANNEL_NAME_4 = "Modeling Paramaters"


SEED = 1234


def run_transformers(
    checkpoint_path: Annotated[
        str,
        Option(help="Path to the hf transformers checkpoint dir."),
    ],
    lighteval_config_path: Annotated[
        str, Option(help="Path to a YAML config to be used for the evaluation.")
    ],
    cache_dir: Annotated[
        str, Option(help="Cache directory for datasets and models.")
    ] = CACHE_DIR,
):
    """
    Evaluate models using nanotron as backend.
    """
    from lighteval.config.lighteval_config import LightEvalConfig
    from lighteval.logging.evaluation_tracker import EvaluationTracker
    from lighteval.logging.hierarchical_logger import htrack_block
    from lighteval.models.base_model import BaseModel, BaseModelConfig
    from lighteval.pipeline import Pipeline, PipelineParameters
    from lighteval.utils.utils import EnvConfig


    env_config = EnvConfig(token=os.getenv("HF_TOKEN"), cache_dir=cache_dir)

    with htrack_block("Load nanotron config"):
        # # Create nanotron config
        # if not checkpoint_path.endswith(".yaml"):
        #     raise ValueError("The checkpoint path should point to a YAML file")

        # model_config = get_config_from_file(
        #     checkpoint_path,
        #     config_class=Config,
        #     model_config_class=None,
        #     skip_unused_config_keys=True,
        #     skip_null_keys=True,
        # )

        # We are getting an type error, because the get_config_from_file is not correctly typed,
        lighteval_config: LightEvalConfig = get_config_from_file(lighteval_config_path, config_class=LightEvalConfig)  # type: ignore
        basemodel_config = BaseModelConfig(checkpoint_path)
        basemodel_config.model_parallel = False
        base_model = BaseModel(env_config=env_config, config=basemodel_config)
        # nanotron_config = FullNanotronConfig(lighteval_config, model_config)

    evaluation_tracker = EvaluationTracker(
        output_dir=lighteval_config.logging.local_output_path,
        # hub_results_org=lighteval_config.logging.results_org,
        # public=lighteval_config.logging.public_run,
        push_to_hub=lighteval_config.logging.push_results_to_hub,
        push_to_tensorboard=lighteval_config.logging.push_results_to_tensorboard,
        # save_details=lighteval_config.logging.save_details,
        tensorboard_metric_prefix=lighteval_config.logging.tensorboard_metric_prefix,
        # nanotron_run_info=nanotron_config.nanotron_config.general,
    )

    pipeline_parameters = PipelineParameters(
        launcher_type=None,
        env_config=env_config,
        job_id=os.environ.get("SLURM_JOB_ID", 0),
        dataset_loading_processes=lighteval_config.tasks.dataset_loading_processes,
        custom_tasks_directory=lighteval_config.tasks.custom_tasks,
        override_batch_size=lighteval_config.batch_size,
        num_fewshot_seeds=1,
        max_samples=lighteval_config.tasks.max_samples,
        use_chat_template=False,
        system_prompt=None,
    )

    pipeline = Pipeline(
        tasks=lighteval_config.tasks.tasks,
        pipeline_parameters=pipeline_parameters,
        evaluation_tracker=evaluation_tracker,
        model=base_model,
    )

    pipeline.evaluate()

    pipeline.show_results()

    pipeline.save_and_push_results()


def create_lighteval_config(
    output_dir: str = "./eval_results",
    tasks: str = "lighteval|agieval:aqua-rat|5|0",
    custom_tasks: str = None,
    batch_size: int = 16,
    dp: int = 1,
    pp: int = 1,
    tp: int = 1,
    max_samples: Optional[int] = None,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: int = 42,
    use_cache: bool = True,
    save_details: bool = True,
    push_to_hub: bool = False,
    results_org: Optional[str] = None,
) -> LightEvalConfig:
    """
    Create a LightEvalConfig object programmatically.

    Args:
        output_dir: Directory where evaluation results will be saved
        tasks: Task specification in format "suite|task|num_few_shots|truncate_few_shots"
        batch_size: Batch size for evaluation
        dp: Data parallel size
        pp: Pipeline parallel size
        tp: Tensor parallel size
        max_samples: Maximum number of samples to evaluate (None for all)
        temperature: Generation temperature
        top_k: Top-k for sampling
        top_p: Top-p for sampling
        seed: Random seed
        use_cache: Whether to use KV cache during generation
        save_details: Whether to save detailed results
        push_to_hub: Whether to push results to Hugging Face Hub
        results_org: Organization to push results to on the Hub

    Returns:
        LightEvalConfig: Config object for lighteval
    """
    # Create logging config
    logging_args = LightEvalLoggingArgs(
        output_dir=output_dir,
        save_details=save_details,
        push_to_hub=push_to_hub,
        push_to_tensorboard=False,
        public_run=False,
        results_org=results_org,
        tensorboard_metric_prefix="eval",
    )

    # Create tasks config
    tasks_args = LightEvalTasksArgs(
        tasks=tasks,
        custom_tasks=custom_tasks,
        max_samples=max_samples,
        dataset_loading_processes=8,
        multichoice_continuations_start_space=None,
        pairwise_tokenization=False,
    )

    # Create parallelism config
    parallelism_args = ParallelismArgs(
        dp=dp,
        pp=pp,
        tp=tp,
    )

    # Create generation config
    generation_args = GenerationArgs(
        sampler="greedy",
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        n_samples=1,
        seed=seed,
        use_cache=use_cache,
    )

    # Return the full config
    return LightEvalConfig(
        logging=logging_args,
        tasks=tasks_args,
        parallelism=parallelism_args,
        batch_size=batch_size,
        generation=generation_args,
    )


def save_lighteval_config_as_yaml(config: LightEvalConfig, output_path: str) -> None:
    """
    Save a LightEvalConfig object as a YAML file.

    Args:
        config: LightEvalConfig object
        output_path: Path to save the YAML file
    """
    # Make directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save config as YAML
    with open(output_path, "w") as f:
        # Create a clean dictionary representation
        config_dict = {
            "logging": {
                "output_dir": config.logging.output_dir,
                "save_details": config.logging.save_details,
                "push_to_hub": config.logging.push_to_hub,
                "push_to_tensorboard": config.logging.push_to_tensorboard,
                "public_run": config.logging.public_run,
                "results_org": config.logging.results_org,
                "tensorboard_metric_prefix": config.logging.tensorboard_metric_prefix,
            },
            "tasks": {
                "tasks": config.tasks.tasks,
                "custom_tasks": config.tasks.custom_tasks,
                "max_samples": config.tasks.max_samples,
                "dataset_loading_processes": config.tasks.dataset_loading_processes,
                "multichoice_continuations_start_space": config.tasks.multichoice_continuations_start_space,
                "pairwise_tokenization": config.tasks.pairwise_tokenization,
            },
            "parallelism": {
                "dp": config.parallelism.dp,
                "pp": config.parallelism.pp,
                "tp": config.parallelism.tp,
            },
            "batch_size": config.batch_size,
            "generation": {
                "sampler": (
                    config.generation.sampler.name.lower()
                    if hasattr(config.generation.sampler, "name")
                    else config.generation.sampler
                ),
                "temperature": config.generation.temperature,
                "top_k": config.generation.top_k,
                "top_p": config.generation.top_p,
                "n_samples": config.generation.n_samples,
                "seed": config.generation.seed,
                "use_cache": config.generation.use_cache,
            },
        }

        # Convert to YAML
        import yaml

        yaml.dump(config_dict, f, default_flow_style=False)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-config-path",
        type=str,
        required=True,
        help="Path to the brr checkpoint YAML or python config file, potentially on S3",
    )
    parser.add_argument(
        "--lighteval-override",
        type=str,
        help="Path to a YAML Lighteval config file for evaluation. Example config: configs/examples/lighteval-config.yaml",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory",
    )

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args, unknowns = parser.parse_known_args()

    if args.lighteval_override is None:
        lighteval_config_path = "configs/examples/lighteval-config.yaml"

        # Create a custom config
        custom_config = create_lighteval_config(
            output_dir="./eval_results/custom",
            tasks="custom|hellaswag|0|1,custom|winogrande|0|1,custom|piqa|0|1,custom|siqa|0|1,custom|openbookqa|0|1,custom|arc:easy|0|1,custom|arc:challenge|0|1,custom|commonsense_qa|0|1,custom|mmlu:abstract_algebra|0|1,custom|mmlu:anatomy|0|1,custom|mmlu:astronomy|0|1,custom|mmlu:business_ethics|0|1,custom|mmlu:clinical_knowledge|0|1,custom|mmlu:college_biology|0|1,custom|mmlu:college_chemistry|0|1,custom|mmlu:college_computer_science|0|1,custom|mmlu:college_mathematics|0|1,custom|mmlu:college_medicine|0|1,custom|mmlu:college_physics|0|1,custom|mmlu:computer_security|0|1,custom|mmlu:conceptual_physics|0|1,custom|mmlu:econometrics|0|1,custom|mmlu:electrical_engineering|0|1,custom|mmlu:elementary_mathematics|0|1,custom|mmlu:formal_logic|0|1,custom|mmlu:global_facts|0|1,custom|mmlu:high_school_biology|0|1,custom|mmlu:high_school_chemistry|0|1,custom|mmlu:high_school_computer_science|0|1,custom|mmlu:high_school_european_history|0|1,custom|mmlu:high_school_geography|0|1,custom|mmlu:high_school_government_and_politics|0|1,custom|mmlu:high_school_macroeconomics|0|1,custom|mmlu:high_school_mathematics|0|1,custom|mmlu:high_school_microeconomics|0|1,custom|mmlu:high_school_physics|0|1,custom|mmlu:high_school_psychology|0|1,custom|mmlu:high_school_statistics|0|1,custom|mmlu:high_school_us_history|0|1,custom|mmlu:high_school_world_history|0|1,custom|mmlu:human_aging|0|1,custom|mmlu:human_sexuality|0|1,custom|mmlu:international_law|0|1,custom|mmlu:jurisprudence|0|1,custom|mmlu:logical_fallacies|0|1,custom|mmlu:machine_learning|0|1,custom|mmlu:management|0|1,custom|mmlu:marketing|0|1,custom|mmlu:medical_genetics|0|1,custom|mmlu:miscellaneous|0|1,custom|mmlu:moral_disputes|0|1,custom|mmlu:moral_scenarios|0|1,custom|mmlu:nutrition|0|1,custom|mmlu:philosophy|0|1,custom|mmlu:prehistory|0|1,custom|mmlu:professional_accounting|0|1,custom|mmlu:professional_law|0|1,custom|mmlu:professional_medicine|0|1,custom|mmlu:professional_psychology|0|1,custom|mmlu:public_relations|0|1,custom|mmlu:security_studies|0|1,custom|mmlu:sociology|0|1,custom|mmlu:us_foreign_policy|0|1,custom|mmlu:virology|0|1,custom|mmlu:world_religions|0|1",
            custom_tasks="/fsx/jason/interleaved/custom_tasks.py",
            batch_size=8,
            dp=1,
            pp=1,
            tp=1,
            max_samples=50,  # Use a small number for testing
            temperature=0.0,
        )

        # Save it to a YAML file
        save_lighteval_config_as_yaml(custom_config, lighteval_config_path)
    else:
        lighteval_config_path = args.lighteval_override

    # nanotron(
    #     checkpoint_config_path=args.checkpoint_config_path,
    #     lighteval_config_path=lighteval_config_path,
    #     cache_dir=args.cache_dir,
    # )
    hf_path = str(Path(args.checkpoint_config_path).parents[1] / 'hf') + "/"
    cmd = [
        "python",
        "scripts/fabai/convert/convert_nanotron_to_hf.py",
        f"--checkpoint_path={Path(args.checkpoint_config_path).parent}",
        f"--save_path={hf_path}"
    ]
    subprocess.run(cmd)

    run_transformers(
        checkpoint_path=hf_path,
        lighteval_config_path=lighteval_config_path,
    )

    rmtree(hf_path)


