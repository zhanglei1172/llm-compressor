import argparse
import base64
import random
import time
from io import BytesIO
from typing import List, Optional, Tuple, Union

import numpy as np
import soundfile
import torch
from accelerate import Accelerator, DistributedType
from easydict import EasyDict
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.evaluator import evaluate
from lmms_eval.evaluator_utils import run_task_tests
from lmms_eval.loggers.evaluation_tracker import EvaluationTracker
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)
from lmms_eval.models.simple.qwen2_5_omni import Qwen2_5_Omni
from lmms_eval.protocol import ChatMessages
from lmms_eval.tasks import TaskManager, get_task_dict
from lmms_eval.utils import get_datetime_str, make_table, simple_parse_args_string
from loguru import logger
from qwen_omni_utils import process_mm_info
from torch.distributed._composable.fsdp import fully_shard, register_fsdp_forward_method
from tqdm import tqdm
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor


def encode_base64(media: tuple[np.ndarray, float]) -> str:
    audio, sr = media

    with BytesIO() as buffer:
        soundfile.write(buffer, audio, sr, format="WAV")
        data = buffer.getvalue()

    return base64.b64encode(data).decode("utf-8")


class Qwen3_Omni(Qwen2_5_Omni):
    is_simple = False

    def __init__(
        self,
        args,
        pretrained: str = "Qwen/Qwen2.5-Omni-7B",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[bool] = "flash_attention_2",
        max_num_frames: int = 768,
        use_custom_video_loader: Optional[bool] = False,
        max_pixels: int = 1605632,
        min_image_pixels=28,
        fps: Optional[
            float
        ] = None,  # Only applicable if use_custom_video_loader is True
        max_image_size: Optional[
            int
        ] = None,  # Only applicable if use_custom_video_loader is True
        **kwargs,
    ) -> None:
        lmms.__init__(self)
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        self.max_pixels = max_pixels
        self.min_pixels = min_image_pixels
        # if self.fps and not self.use_custom_video_loader:
        #     raise ValueError("FPS is only applicable if use_custom_video_loader is True")
        self.max_image_size = max_image_size
        if self.max_image_size and not self.use_custom_video_loader:
            raise ValueError(
                "max_image_size is only applicable if use_custom_video_loader is True"
            )

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        Qwen3OmniMoeForConditionalGeneration._tp_plan = (
            []
            if Qwen3OmniMoeForConditionalGeneration._tp_plan is None
            else Qwen3OmniMoeForConditionalGeneration._tp_plan
        )
        self._model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            pretrained,
            torch_dtype="auto",
            device_map=self.device_map,
            attn_implementation=attn_implementation,
        ).eval()
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(pretrained)
        self.max_num_frames = max_num_frames
        self._tokenizer = self.processor.tokenizer

        self._config = self._model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self._model.disable_talker()
        self._reform_model(args)

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(
                    self._model,
                    torch.optim.SGD(self._model.thinker.lm_head.parameters()),
                )[0]
                register_fsdp_forward_method(self._model, "generate")
            else:
                self._model = accelerator.prepare_model(
                    self._model, evaluation_mode=True
                )
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                logger.info(
                    f"Using {accelerator.num_processes} devices with data parallelism"
                )
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    def _reform_model(self, args):
        if args.replace_aut:
            from llmcompressor.modeling.qwen3_omni_moe import replace_rmsnorm

            replace_rmsnorm(self._model.thinker.audio_tower)
            tensor = self._model.thinker.audio_tower.positional_embedding.positional_embedding
            ori_device = tensor.device
            ori_shape = tensor.shape
            ori_dtype = tensor.dtype
            Q1 = torch.load(f"{args.model_path}/transform_state_dict.pt")[
                "audio_tower.positional_embedding.R1_weight_output"
            ]["weight"].to(dtype=torch.float64, device=ori_device)
            self._model.thinker.audio_tower.positional_embedding.positional_embedding = (
                (
                    (tensor - tensor.mean(-1, keepdim=True))
                    .to(dtype=Q1.dtype)
                    .reshape(-1, ori_shape[-1] // Q1.shape[0], Q1.shape[0])
                    @ Q1
                )
                .to(dtype=ori_dtype, device=ori_device)
                .reshape(ori_shape)
            )
        if args.replace_vit:
            from llmcompressor.modeling.qwen3_omni_moe import replace_rmsnorm

            replace_rmsnorm(self._model.thinker.visual)

    @property
    def model(self):
        # # returns the model, unwrapping it if using Accelerate
        # if hasattr(self, "accelerator"):
        #     return self.accelerator.unwrap_model(self._model)
        # else:
        return self._model

    def _patch_audio(self, conversations: List[List[dict]]):
        for conversation in conversations:
            for message in conversation:
                if not isinstance(message["content"], list):
                    continue
                for ele in message["content"]:
                    if ele["type"] == "audio":
                        if "audio" in ele:
                            key = "audio"
                        elif "audio_url" in ele:
                            key = "audio_url"
                        else:
                            continue
                        path = ele.get(key)
                        if isinstance(path, dict):
                            sampling_rate = path.get("sampling_rate", 16000)
                            array = path.get("array", None)
                            ele[key] = (
                                f"data:audio/wav;base64,{encode_base64((array, sampling_rate))}"
                            )

    @torch.no_grad
    def generate_until(self, requests: List[Instance]) -> List[str]:
        current_use_audio = False
        res = []

        # A dummy collate here to sort by doc id
        def _collate(x):
            return x[0], x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator(
            [reg.args for reg in requests],
            _collate,
            group_fn=lambda x: x[2],
            grouping=True,
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = (
            len(requests) // self.batch_size
            if len(requests) % self.batch_size == 0
            else len(requests) // self.batch_size + 1
        )
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        e2e_latency = 0
        total_tokens = 0
        for chunk in chunks:
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            chat_messages = [
                doc_to_messages[idx](self.task_dict[task][split][ids])
                for idx, (ids, task, split) in enumerate(zip(doc_id, task, split))
            ]
            chat_messages: List[ChatMessages] = [
                ChatMessages(**{"messages": message}) for message in chat_messages
            ]
            visuals = []
            videos = []
            for messages in chat_messages:
                visual, video, _ = messages.extract_media()
                visuals.append(visual)
                videos.append(video)
            visuals = self.flatten(visuals)
            videos = self.flatten(videos)
            gen_kwargs = all_gen_kwargs[0]

            # Apply chat template
            video_kwargs = {
                "max_pixels": self.max_pixels,
                "min_pixels": self.min_pixels,
            }
            if self.fps is not None:
                video_kwargs["fps"] = self.fps
            else:
                video_kwargs["nframes"] = self.max_num_frames
            batched_messages = [
                chat_message.to_hf_messages(video_kwargs=video_kwargs)
                for chat_message in chat_messages
            ]
            self._patch_audio(batched_messages)
            texts = [
                self.processor.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=True
                )
                for msg in batched_messages
            ]
            audios_inputs, image_inputs, video_inputs = process_mm_info(
                batched_messages, use_audio_in_video=current_use_audio
            )
            if video_inputs is not None:
                total_frames = video_inputs[0].shape[0]
                indices = np.linspace(
                    0, total_frames - 1, self.max_num_frames, dtype=int
                )
                # Append the last frame index if not already included
                if total_frames - 1 not in indices:
                    indices = np.append(indices, total_frames - 1)
                video_inputs[0] = video_inputs[0][indices]
            inputs = self.processor(
                text=texts,
                audio=audios_inputs,
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=current_use_audio,
            )

            if self.device_map == "auto":
                inputs = inputs.to(device="cuda", dtype=self.model.dtype)
            else:
                inputs = inputs.to(device=self.device, dtype=self.model.dtype)

            # Set default generation kwargs
            default_gen_kwargs = {
                "max_new_tokens": 128,
                "temperature": 0.0,  # Set to 0 for greedy default
                "top_p": None,
                "num_beams": 1,
            }
            # Update with provided kwargs
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id

            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None
                current_gen_kwargs["top_k"] = None

            start_time = time.time()
            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=current_gen_kwargs["do_sample"],
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                top_k=current_gen_kwargs.get("top_k", None),
                use_cache=self.use_cache,
            )[0]  # TODO only for omni
            end_time = time.time()

            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, cont)
            ]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            # Calculate timing metrics for batch
            e2e_latency += end_time - start_time
            total_tokens += sum(len(ids) for ids in generated_ids_trimmed)

            for ans, context in zip(answers, texts):
                clean_ans = parse_reasoning_model_answer(ans)
                res.append(clean_ans)
                self.cache_hook.add_partial(
                    "generate_until", (context, gen_kwargs), clean_ans
                )
                pbar.update(1)

                logger.debug(f"Question: {context}")
                logger.debug(f"Model Raw Response: {ans}")
                logger.debug(f"Model Clean Response: {clean_ans}")
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        # Calculate average speed
        avg_speed = total_tokens / e2e_latency if e2e_latency > 0 else 0
        # Log metrics
        metric_dict = {
            "total_tokens": total_tokens,
            "e2e_latency": e2e_latency,
            "avg_speed": avg_speed,
            "additional_metrics": {
                "rank": self.rank,
            },
        }
        log_metrics(**metric_dict)

        pbar.close()
        return res


def get_model(args, model_args, batch_size, device):
    lm = Qwen3_Omni.create_from_arg_string(
        model_args,
        {
            "batch_size": batch_size,
            "device": device,
            "args": args,
        },
    )
    return lm


def eval(
    args,
    model_args: Optional[Union[str, dict]] = None,
    tasks: Optional[List[Union[str, dict, object]]] = None,
    num_fewshot: Optional[int] = None,
    device: Optional[str] = None,
    use_cache: Optional[str] = None,
    cache_requests: bool = False,
    rewrite_requests_cache: bool = False,
    limit: Optional[Union[int, float]] = None,
    bootstrap_iters: int = 100000,
    check_integrity: bool = False,
    write_out: bool = False,
    log_samples: bool = True,
    evaluation_tracker: Optional[EvaluationTracker] = None,
    system_instruction: Optional[str] = None,
    apply_chat_template: bool = False,
    fewshot_as_multiturn: bool = False,
    gen_kwargs: Optional[str] = None,
    task_manager: Optional[TaskManager] = None,
    verbosity: str = "INFO",
    predict_only: bool = False,
    random_seed: int = 0,
    numpy_random_seed: int = 1234,
    torch_random_seed: int = 1234,
    fewshot_random_seed: int = 1234,
    datetime_str: str = get_datetime_str(),
    cli_args=EasyDict({"output_path": "./out", "process_with_media": None}),
):
    limit = args.eval_limit if args.eval_limit is not None else limit
    model_args = "pretrained=" + args.model_path + ",device_map=auto"
    batch_size = args.eval_bs
    tasks = args.eval_dataset_name
    num_fewshot = 0

    seed_message = []
    if random_seed is not None:
        # See https://github.com/EleutherAI/lm-evaluation-harness/pull/1412
        seed_message.append(f"Setting random seed to {random_seed}")
        random.seed(random_seed)

    if numpy_random_seed is not None:
        seed_message.append(f"Setting numpy seed to {numpy_random_seed}")
        np.random.seed(numpy_random_seed)

    if torch_random_seed is not None:
        seed_message.append(f"Setting torch manual seed to {torch_random_seed}")
        torch.manual_seed(torch_random_seed)

    if seed_message:
        logger.info(" | ".join(seed_message))

    assert (
        tasks != []
    ), "No tasks specified, or no tasks found. Please verify the task names."

    if gen_kwargs:
        gen_kwargs = simple_parse_args_string(gen_kwargs)
        logger.warning("generation_kwargs specified through cli.")
        if gen_kwargs == "":
            gen_kwargs = None

    if model_args is None:
        model_args = ""

    if task_manager is None:
        task_manager = TaskManager(verbosity, model_name=args.model_path, include_path='/workspace/zhangl98@xiaopeng.com/code/llm-compressor/debug/openasr')

    task_dict = get_task_dict(tasks, task_manager, task_type="chat")

    lm = get_model(args, model_args, batch_size, device)

    def _adjust_config(task_dict):
        adjusted_task_dict = {}
        for task_name, task_obj in task_dict.items():
            if isinstance(task_obj, dict):
                adjusted_task_dict = {
                    **adjusted_task_dict,
                    **{task_name: _adjust_config(task_obj)},
                }

            else:
                task_obj = task_dict[task_name]
                if isinstance(task_obj, tuple):
                    group, task_obj = task_obj
                    if task_obj is None:
                        continue
                lm.task_dict[task_name] = task_obj.dataset
                if "generate_until" in task_obj.get_config("output_type"):
                    if gen_kwargs is not None:
                        task_obj.set_config(
                            key="generation_kwargs", value=gen_kwargs, update=True
                        )

                if predict_only:
                    logger.info(f"Processing {task_name} in output-only mode. \
                                Metrics will not be calculated!")
                    # we have to change the class properties post-hoc. This is pretty hacky.
                    task_obj.override_metric(metric_name="bypass")

                # override tasks' fewshot values to
                # the provided num_fewshot arg value
                # except if tasks have it set to 0 manually in their configs--then
                # we should never overwrite that
                if num_fewshot is not None:
                    if (default_num_fewshot := task_obj.get_config("num_fewshot")) == 0:
                        logger.info(f"num_fewshot has been set to 0 for {task_name} \
                                    in its config. Manual configuration will be ignored.")
                    else:
                        logger.warning(f"Overwriting default num_fewshot of {task_name} \
                                        from {default_num_fewshot} to {num_fewshot}")
                        task_obj.set_config(key="num_fewshot", value=num_fewshot)
                else:
                    # if num_fewshot not provided, and the task does not define a default one,
                    # default to 0
                    if (
                        default_num_fewshot := task_obj.get_config("num_fewshot")
                    ) is None:
                        task_obj.set_config(key="num_fewshot", value=0)
                # fewshot_random_seed set for tasks, even with a default num_fewshot
                # (e.g. in the YAML file)
                task_obj.set_fewshot_seed(seed=fewshot_random_seed)
                # logger.info(f"Setting fewshot random generator seed to {fewshot_random_seed}")

                adjusted_task_dict[task_name] = task_obj

        return adjusted_task_dict

    task_dict = _adjust_config(task_dict)

    if check_integrity:
        run_task_tests(task_list=tasks)

    if evaluation_tracker is not None:
        evaluation_tracker.general_config_tracker.log_experiment_args(
            model_source=args.model_path,
            model_args=model_args,
            system_instruction=system_instruction,
            chat_template=lm.chat_template if apply_chat_template else None,
            fewshot_as_multiturn=fewshot_as_multiturn,
        )

    results = evaluate(
        lm=lm,
        task_dict=task_dict,
        limit=limit,
        cache_requests=cache_requests,
        rewrite_requests_cache=rewrite_requests_cache,
        bootstrap_iters=bootstrap_iters,
        write_out=write_out,
        log_samples=True if predict_only else log_samples,
        system_instruction=system_instruction,
        apply_chat_template=apply_chat_template,
        fewshot_as_multiturn=fewshot_as_multiturn,
        verbosity=verbosity,
        cli_args=cli_args,
    )

    if hasattr(lm, "_model"):
        del lm._model
        torch.cuda.empty_cache()

    if lm.rank == 0:
        model_name = args.model_path

        # add info about the model and few shot config
        results["config"] = {
            "model": model_name,
            "model_args": model_args,
        }
        # add more detailed model info if available TODO: add model info
        # if isinstance(lm, lm_eval.models.huggingface.HFLM):
        #     results["config"].update(lm.get_model_info())
        # add info about execution
        results["config"].update(
            {
                "batch_size": batch_size,
                "batch_sizes": (
                    list(lm.batch_sizes.values()) if hasattr(lm, "batch_sizes") else []
                ),
                "device": device,
                "use_cache": use_cache,
                "limit": limit,
                "bootstrap_iters": bootstrap_iters,
                "gen_kwargs": gen_kwargs,
                "random_seed": random_seed,
                "numpy_seed": numpy_random_seed,
                "torch_seed": torch_random_seed,
                "fewshot_seed": fewshot_random_seed,
            }
        )
        results["date"] = datetime_str
        # add_env_info(results)  # additional environment info to results
        # add_tokenizer_info(results, lm)  # additional info about tokenizer
        return "\n" + make_table(results)
    else:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--eval_dataset_name", type=str, default="mmmu")
    parser.add_argument("--eval_limit", type=int, required=False)
    parser.add_argument("--eval_bs", type=int, required=False)
    parser.add_argument("--replace_aut", action="store_true")
    parser.add_argument("--replace_vit", action="store_true")
    args = parser.parse_args()
    ret = eval(args)
    if ret:
        print(args.model_path)
        print(ret)
