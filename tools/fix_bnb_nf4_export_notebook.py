"""Patch the DiffusionGemma NF4 export notebook to keep full bnb QuantState.

The original _True14GB_BNB_NF4_Export.ipynb saved an empty
`quant_state.bitsandbytes__nf4` tensor, discarding blocksize / shape / dtype /
nested_offset metadata. Downstream engines then cannot recover the nested
mean-offset that bitsandbytes uses for double quant.

This script rewrites the export cell to save `QuantState.as_dict(packed=True)`,
which packs all non-tensor metadata as JSON bytes into
`<name>.quant_state.bitsandbytes__nf4` (the format bnb itself uses), and adds a
verification cell that prints one tensor's recovered QuantState.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


DEFAULT_NOTEBOOK = Path(
    r"D:\Downloads\DiffusionGemma_True14GB_BNB_NF4_Export.ipynb"
)

OLD_BLOCK = """    quant_state_dict[name] = quant_w.cpu()
    quant_state_dict[f"{name}.absmax"] = qstate.absmax.cpu()
    quant_state_dict[f"{name}.quant_map"] = qstate.code.cpu()
    quant_state_dict[f"{name}.quant_state.bitsandbytes__nf4"] = torch.tensor([], dtype=torch.uint8)

    if qstate.nested:
        quant_state_dict[f"{name}.nested_absmax"] = qstate.state2.absmax.cpu()
        quant_state_dict[f"{name}.nested_quant_map"] = qstate.state2.code.cpu()"""

NEW_BLOCK = """    # 保存完整 QuantState（含 packed 元数据: blocksize/shape/dtype/nested_offset 等），
    # 以便下游引擎可以直接从 quant_state.bitsandbytes__nf4 恢复，而不需要猜测 offset。
    quant_state_dict[name] = quant_w.cpu()
    packed_qs = qstate.as_dict(packed=True)
    for qs_key, qs_tensor in packed_qs.items():
        quant_state_dict[f"{name}.{qs_key}"] = qs_tensor.cpu()"""

VERIFY_CELL = """# 校验: quant_state.bitsandbytes__nf4 必须包含完整元数据（非空）
from bitsandbytes.utils import unpack_tensor_to_dict

with safe_open(export_dir / "model.safetensors", framework="pt", device="cpu") as f:
    sample = next(
        k for k in f.keys()
        if k.endswith(".quant_state.bitsandbytes__nf4")
    )
    blob = f.get_tensor(sample)
    print("sample key:", sample)
    print("quant_state blob bytes:", blob.numel())
    meta = unpack_tensor_to_dict(blob)
    print("recovered metadata:", json.dumps(meta, indent=2))
    assert blob.numel() > 0, "quant_state metadata must not be empty"
    assert "nested_offset" in meta, "nested_offset missing from quant_state"
    assert meta["blocksize"] == 64, "unexpected blocksize"
    print("QuantState metadata verification PASSED.")"""


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_NOTEBOOK
    if not path.exists():
        print(f"notebook not found: {path}", file=sys.stderr)
        return 1

    backup = path.with_suffix(".orig.ipynb")
    if not backup.exists():
        shutil.copyfile(path, backup)
        print(f"backup written: {backup}")

    notebook = json.loads(path.read_text(encoding="utf-8"))
    target = None
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell.get("source", []))
        if (
            cell.get("cell_type") == "code"
            and "quant_state_dict[name] = quant_w.cpu()" in source
        ):
            target = index
            break
    if target is None:
        print("export cell not found", file=sys.stderr)
        return 1

    source = "".join(notebook["cells"][target].get("source", []))
    if OLD_BLOCK not in source:
        print("old export block not found (already patched?)", file=sys.stderr)
        return 1
    notebook["cells"][target]["source"] = source.replace(
        OLD_BLOCK, NEW_BLOCK
    ).splitlines(keepends=True)

    notebook["cells"].append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": VERIFY_CELL.splitlines(keepends=True),
        }
    )

    path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"patched: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
