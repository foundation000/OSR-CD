from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
import sys
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from osrcd.data import Triple, UnifiedSample


os.environ.setdefault("DGLBACKEND", "pytorch")
os.environ.setdefault("DGLDEFAULTDIR", str(ROOT / "preprocess" / ".dgl"))


def to_python(value: Any) -> Any:
    """Convert common tensor/array/scalar values into JSON-friendly objects."""
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def label_to_text(label: Any) -> str:
    return json.dumps(to_python(label), ensure_ascii=False)


def load_pickle_dataset(path: str | Path) -> list:
    try:
        with Path(path).open("rb") as f:
            return pickle.load(f)
    except ModuleNotFoundError as exc:
        if exc.name == "dgl":
            raise RuntimeError(
                "This dataset stores pickled DGLGraph objects and requires DGL to read. "
                "Install a DGL build compatible with this Python environment, then rerun the adapter."
            ) from exc
        raise


def load_index_file(path: str | Path) -> List[int]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [int(x) for x in json.load(f)]


def load_split_lookup(data_dir: str | Path) -> Dict[int, str]:
    data_dir = Path(data_dir)
    lookup: Dict[int, str] = {}
    for split in ["train", "val", "test"]:
        path = data_dir / f"{split}_index.json"
        if not path.exists():
            continue
        for idx in load_index_file(path):
            lookup[idx] = split
    return lookup


def load_smiles(data_dir: str | Path) -> List[str]:
    data_dir = Path(data_dir)
    for name in ["smiles_1_direct.txt", "smiles_2_explicit_h.txt", "smiles_3_addhs.txt", "smiles_4_addhs_explicit_h.txt"]:
        path = data_dir / name
        if path.exists():
            return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return []


def raw_data_dir(dataset: str) -> Path:
    return ROOT / "preprocess" / "raw_data" / dataset


def stage1_output_path(dataset: str) -> Path:
    return ROOT / "preprocess" / "stage1" / dataset / "samples.jsonl"


def _flat_preview(value: Any, max_items: int = 8) -> str:
    value = to_python(value)
    if isinstance(value, list):
        flat: List[Any] = []

        def collect(obj: Any) -> None:
            if len(flat) >= max_items:
                return
            if isinstance(obj, list):
                for item in obj:
                    collect(item)
                    if len(flat) >= max_items:
                        break
            else:
                flat.append(obj)

        collect(value)
        suffix = ", ..." if len(flat) >= max_items else ""
        return "[" + ", ".join(str(x) for x in flat[:max_items]) + suffix + "]"
    return str(value)


def _feature_at(container: Dict[str, Any], key: str, idx: int) -> Any:
    value = container[key]
    try:
        return value[idx]
    except Exception:
        return value


def _scalar_feature(container: Dict[str, Any], keys: Iterable[str], idx: int) -> Optional[int]:
    for key in keys:
        if key not in container:
            continue
        value = to_python(_feature_at(container, key, idx))
        while isinstance(value, list) and value:
            value = value[0]
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def _describe_features(container: Dict[str, Any], idx: int, preferred_keys: Iterable[str]) -> str:
    parts: List[str] = []
    for key in preferred_keys:
        if key not in container:
            continue
        parts.append(f"{key}={_flat_preview(_feature_at(container, key, idx))}")
    return ", ".join(parts)


ELEMENTS = {
    1: ("hydrogen", "H"),
    3: ("lithium", "Li"),
    5: ("boron", "B"),
    6: ("carbon", "C"),
    7: ("nitrogen", "N"),
    8: ("oxygen", "O"),
    9: ("fluorine", "F"),
    11: ("sodium", "Na"),
    12: ("magnesium", "Mg"),
    13: ("aluminum", "Al"),
    14: ("silicon", "Si"),
    15: ("phosphorus", "P"),
    16: ("sulfur", "S"),
    17: ("chlorine", "Cl"),
    19: ("potassium", "K"),
    20: ("calcium", "Ca"),
    25: ("manganese", "Mn"),
    26: ("iron", "Fe"),
    27: ("cobalt", "Co"),
    28: ("nickel", "Ni"),
    29: ("copper", "Cu"),
    30: ("zinc", "Zn"),
    33: ("arsenic", "As"),
    35: ("bromine", "Br"),
    53: ("iodine", "I"),
}

BOND_TYPES = {
    0: "single molecular bond",
    1: "single molecular bond",
    2: "double molecular bond",
    3: "triple molecular bond",
    4: "aromatic molecular bond",
}

ZERO_BASED_BOND_TYPES = {
    0: "single molecular bond",
    1: "double molecular bond",
    2: "triple molecular bond",
    3: "aromatic molecular bond",
}

HYBRIDIZATION_TYPES = {
    0: "sp hybridization",
    1: "sp2 hybridization",
    2: "sp3 hybridization",
    3: "sp3d hybridization",
    4: "sp3d2 hybridization",
}

MOLECULAR_DGL_DATASETS = {"molhiv", "qm9", "zinc"}
MOLECULAR_LIGHTWEIGHT_DATASETS = {"p_func", "p_struct"}


def _element_text(atomic_number: int, fallback: str = "atom") -> str:
    name, symbol = ELEMENTS.get(atomic_number, (fallback, str(atomic_number)))
    return f"{name} atom ({symbol})"


def _as_int_list(value: Any) -> List[int]:
    value = to_python(value)
    if isinstance(value, list):
        out: List[int] = []

        def collect(obj: Any) -> None:
            if isinstance(obj, list):
                for item in obj:
                    collect(item)
            else:
                try:
                    out.append(int(round(float(obj))))
                except (TypeError, ValueError):
                    pass

        collect(value)
        return out
    try:
        return [int(round(float(value)))]
    except (TypeError, ValueError):
        return []


def _vector_feature(container: Dict[str, Any], keys: Iterable[str], idx: int) -> List[int]:
    for key in keys:
        if key in container:
            values = _as_int_list(_feature_at(container, key, idx))
            if values:
                return values
    return []


def _qm9_node_text(node_id: Any, features: Dict[str, Any]) -> Optional[str]:
    values = _vector_feature(features, ["attr"], int(node_id))
    if len(values) >= 6:
        return f"atom_{node_id} {_element_text(values[5])}"
    return None


def _qm9_edge_text(edge_idx: int, features: Dict[str, Any]) -> Optional[str]:
    values = _vector_feature(features, ["edge_attr"], edge_idx)
    if values:
        try:
            return ZERO_BASED_BOND_TYPES.get(values.index(1), "molecular bond")
        except ValueError:
            return None
    return None


def _atomic_number_node_text(node_id: Any, features: Dict[str, Any]) -> Optional[str]:
    atomic_number = _scalar_feature(features, ["node_type_id", "feat"], int(node_id))
    if atomic_number is None:
        return None
    return f"atom_{node_id} {_element_text(atomic_number)}"


def _bond_type_edge_text(edge_idx: int, features: Dict[str, Any]) -> Optional[str]:
    bond_type = _scalar_feature(features, ["edge_type_id", "feat"], edge_idx)
    if bond_type is None:
        return None
    return BOND_TYPES.get(bond_type, f"molecular bond type {bond_type}")


def _peptide_node_text(node_id: Any, values: Sequence[int]) -> str:
    atomic_number = values[0] + 1 if values else 0
    parts = [f"atom_{node_id} {_element_text(atomic_number)}"]
    if len(values) >= 3:
        parts.append(f"degree {values[2]}")
    if len(values) >= 4:
        charge = values[3] - 5
        if charge:
            parts.append(f"formal charge {charge:+d}")
    if len(values) >= 5 and values[4]:
        parts.append(f"{values[4]} attached hydrogens")
    if len(values) >= 7 and values[6] in HYBRIDIZATION_TYPES:
        parts.append(HYBRIDIZATION_TYPES[values[6]])
    if len(values) >= 8 and values[7]:
        parts.append("aromatic atom")
    if len(values) >= 9 and values[8]:
        parts.append("ring atom")
    return ", ".join(parts)


def _peptide_edge_text(values: Sequence[int]) -> str:
    bond_type = ZERO_BASED_BOND_TYPES.get(values[0], "molecular bond") if values else "molecular bond"
    parts = [bond_type]
    if len(values) >= 2 and values[1]:
        parts.append(f"stereochemistry type {values[1]}")
    if len(values) >= 3 and values[2]:
        parts.append("conjugated bond")
    return ", ".join(parts)


def _semantic_node_text(dataset: str, node_id: Any, features: Dict[str, Any]) -> Optional[str]:
    if dataset == "qm9":
        return _qm9_node_text(node_id, features)
    if dataset in {"molhiv", "zinc"}:
        return _atomic_number_node_text(node_id, features)
    node_type = _scalar_feature(features, ["node_type_id", "node_labels", "node_token_ids"], int(node_id))
    if node_type is None:
        return None
    return None


def _semantic_edge_text(dataset: str, edge_idx: int, features: Dict[str, Any]) -> Optional[str]:
    if dataset == "qm9":
        return _qm9_edge_text(edge_idx, features)
    if dataset in {"molhiv", "zinc"}:
        return _bond_type_edge_text(edge_idx, features)
    edge_type = _scalar_feature(features, ["edge_type_id", "edge_labels", "edge_token_ids"], edge_idx)
    if edge_type is None:
        return None
    return None


def dgl_graph_to_triples(graph: Any, dataset: str) -> List[Triple]:
    src, dst = graph.edges()
    src_ids = to_python(src)
    dst_ids = to_python(dst)
    ndata = graph.ndata
    edata = graph.edata
    node_keys = ["node_token_ids", "node_type_id", "node_labels", "node_attr", "feat", "attr", "pos", "_ID"]
    edge_keys = ["edge_token_ids", "edge_type_id", "edge_labels", "feat", "edge_attr", "_ID"]

    triples: List[Triple] = []
    seen_bonds = set()
    for edge_idx, (u, v) in enumerate(zip(src_ids, dst_ids)):
        head = _semantic_node_text(dataset, u, ndata)
        tail = _semantic_node_text(dataset, v, ndata)
        relation = _semantic_edge_text(dataset, edge_idx, edata)
        if head is None:
            head_features = _describe_features(ndata, int(u), node_keys)
            head = f"{dataset} node_{u}" + (f" with {head_features}" if head_features else "")
        if tail is None:
            tail_features = _describe_features(ndata, int(v), node_keys)
            tail = f"{dataset} node_{v}" + (f" with {tail_features}" if tail_features else "")
        if relation is None:
            edge_features = _describe_features(edata, edge_idx, edge_keys)
            relation = f"{dataset} edge_{edge_idx}" + (f" with {edge_features}" if edge_features else "")
        if dataset in MOLECULAR_DGL_DATASETS:
            bond_key = (frozenset((int(u), int(v))), relation)
            if bond_key in seen_bonds:
                continue
            seen_bonds.add(bond_key)
        triples.append((head, relation, tail))
    return triples


def lightweight_graph_to_triples(graph: Dict[str, Any], dataset: str) -> List[Triple]:
    src_ids, dst_ids = graph["edges"]
    src_ids = to_python(src_ids)
    dst_ids = to_python(dst_ids)
    node_data = {
        "node_token_ids": graph.get("node_token_ids"),
        "node_type_id": graph.get("node_type_ids"),
        "node_features": graph.get("node_features"),
    }
    edge_data = {
        "edge_token_ids": graph.get("edge_token_ids"),
        "edge_type_id": graph.get("edge_type_ids"),
        "edge_features": graph.get("edge_features"),
    }
    node_data = {k: v for k, v in node_data.items() if v is not None}
    edge_data = {k: v for k, v in edge_data.items() if v is not None}

    triples: List[Triple] = []
    seen_bonds = set()
    for edge_idx, (u, v) in enumerate(zip(src_ids, dst_ids)):
        if dataset in MOLECULAR_LIGHTWEIGHT_DATASETS:
            head = _peptide_node_text(u, _vector_feature(node_data, ["node_features"], int(u)))
            tail = _peptide_node_text(v, _vector_feature(node_data, ["node_features"], int(v)))
            relation = _peptide_edge_text(_vector_feature(edge_data, ["edge_features"], edge_idx))
            bond_key = (frozenset((int(u), int(v))), relation)
            if bond_key in seen_bonds:
                continue
            seen_bonds.add(bond_key)
        else:
            head_features = _describe_features(node_data, int(u), node_data.keys())
            tail_features = _describe_features(node_data, int(v), node_data.keys())
            edge_features = _describe_features(edge_data, edge_idx, edge_data.keys())
            head = f"{dataset} node_{u}" + (f" with {head_features}" if head_features else "")
            tail = f"{dataset} node_{v}" + (f" with {tail_features}" if tail_features else "")
            relation = f"{dataset} edge_{edge_idx}" + (f" with {edge_features}" if edge_features else "")
        triples.append((head, relation, tail))
    return triples


def graph_to_triples(graph: Any, dataset: str) -> List[Triple]:
    if isinstance(graph, dict):
        return lightweight_graph_to_triples(graph, dataset)
    return dgl_graph_to_triples(graph, dataset)


def serialize_graph_raw(triples: List[Triple]) -> str:
    return " ".join(f"({h}; {r}; {t})" for h, r, t in triples)


def write_samples_jsonl(samples: Iterable[UnifiedSample], output: str | Path) -> int:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_json(), ensure_ascii=False) + "\n")
            count += 1
    return count


def make_pkl_samples(
    *,
    dataset: str,
    input_path: str | Path,
    query_builder: Callable[[int, Any, Dict[str, Any]], str],
    metadata_builder: Optional[Callable[[int, Any, Any, Dict[str, Any]], Dict[str, Any]]] = None,
    split_lookup: Optional[Dict[int, str]] = None,
    limit: Optional[int] = None,
) -> Iterator[UnifiedSample]:
    data = load_pickle_dataset(input_path)
    split_lookup = split_lookup or {}
    for idx, (graph, label) in enumerate(data):
        if limit is not None and idx >= limit:
            break
        context = {"split": split_lookup.get(idx)}
        triples = graph_to_triples(graph, dataset)
        metadata = {
            "source_index": idx,
            "split": split_lookup.get(idx),
            "num_triples": len(triples),
            "label_raw": to_python(label),
        }
        if metadata_builder is not None:
            metadata.update(metadata_builder(idx, graph, label, context))
        answer = label_to_text(label)
        yield UnifiedSample(
            sample_id=idx,
            dataset=dataset,
            query=query_builder(idx, label, context),
            answer=answer,
            label=answer,
            graph_raw=serialize_graph_raw(triples),
            triples=triples,
            metadata=metadata,
        )


def common_parser(description: str, default_input: Path, default_output: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input", type=str, default=str(default_input))
    parser.add_argument("--output", type=str, default=str(default_output))
    parser.add_argument("--limit", type=int, default=None)
    return parser
