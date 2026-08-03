import argparse
import json
import logging
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config

config.setup_env()

import torch
import transformers

import jlens


class ProgressHandler(logging.Handler):
    def emit(self, record):
        if not record.args:
            return
        if record.msg.startswith("  prompt"):
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "done": record.args[0],
                        "total": record.args[1],
                        "seconds": record.args[4],
                    }
                ),
                flush=True,
            )
        elif record.msg.startswith("  resuming"):
            print(
                json.dumps(
                    {"event": "resume", "done": record.args[0], "total": record.args[1]}
                ),
                flush=True,
            )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--quant", default=None)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dim-batch", type=int, required=True)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--source-layers", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, handlers=[ProgressHandler()])

    prompts = json.loads(pathlib.Path(args.prompts).read_text(encoding="utf-8"))
    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    kwargs = {"dtype": torch_dtype, "device_map": {"": args.device}}
    model_source = args.model
    if args.quant == "int8":
        kwargs["quantization_config"] = transformers.BitsAndBytesConfig(load_in_8bit=True)
    elif args.quant == "nf4":
        kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )

    print(json.dumps({"event": "loading", "model": args.model, "device": args.device}), flush=True)
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(model_source, **kwargs)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_source)
    model = jlens.from_hf(hf_model, tokenizer)

    source_layers = json.loads(args.source_layers) if args.source_layers else None
    # large models: the checkpoint (n_layers × d_model² × 4 B) can weigh hundreds
    # of MB — writing it after every prompt would wear the SSD for nothing. We
    # space it out to target ~150 MB of average writes per prompt.
    n_src = len(source_layers) if source_layers else model.n_layers - 1
    ckpt_bytes = n_src * model.d_model**2 * 4
    checkpoint_every = max(1, round(ckpt_bytes / 150e6))
    print(
        json.dumps({"event": "fitting", "total": len(prompts)}),
        flush=True,
    )
    lens = jlens.fit(
        model,
        prompts,
        source_layers=source_layers,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        checkpoint_path=args.checkpoint,
        checkpoint_every=checkpoint_every,
    )
    lens.save(args.out)
    print(json.dumps({"event": "done", "out": args.out, "n_prompts": lens.n_prompts}), flush=True)


if __name__ == "__main__":
    main()
