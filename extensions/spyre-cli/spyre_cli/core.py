from pathlib import Path
from spyre_cli.iospec.iospec import parse_json_file


def _launch(path, tensors):
    # delayed import, for easy testing
    import torch

    spyrecode_dir = path / "spyreCodeDir"
    if not spyrecode_dir.exists():
        raise FileNotFoundError(
            f"spyreCodeDir not found at '{spyrecode_dir}'. "
            "Ensure the model has been compiled before launching."
        )

    # this is the magic line
    # should have already compiled at this point
    jobplan = torch.spyre.prepare_kernel(str(spyrecode_dir))
    torch.spyre.launch_jobplan(jobplan, tensors)


def launch_from_iofile(path=".", iofile="io.json"):
    path = Path(path)
    try:
        spec = parse_json_file(iofile)
    except Exception as e:
        print(e)
        exit(1)

    import torch

    tensors = []
    for tensor_spec in spec.inputs + spec.outputs:
        dtype = getattr(torch, tensor_spec.dtype.value)
        tensor = torch.ones(
            tensor_spec.dimensions,
            dtype=dtype,
            device="spyre",
        )
        tensors.append(tensor)

    _launch(path, tensors)

    print(tensors[-1])


def launch(*args, **kwargs):
    path = Path(kwargs.get("path", "."))

    _launch(path, args)
