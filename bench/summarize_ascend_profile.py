"""Extract the relevant layer-0 decode operators from an msprof export."""
import argparse
import csv
import json
from pathlib import Path


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir", help="ASCEND_PROFILER_OUTPUT directory")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.output_dir)
    stats = rows(root / "op_statistic.csv")
    wanted = ("MatMul", "TransData", "Mul", "Slice", "Conv", "Gdn", "Copy", "Cast", "Reduce")
    selected = [r for r in stats if any(k.lower() in r["OP Type"].lower() for k in wanted)]
    api = rows(root / "api_statistic.csv")
    selected_api = [r for r in api if any(k.lower() in r["API Name"].lower() for k in wanted)]
    kernels = rows(root / "kernel_details.csv")
    custom = [r for r in kernels if "GdnConvStateUpdate" in r.get("Name", "")]
    report = {"source": str(root), "operator_stats": selected,
              "api_stats": selected_api, "custom_kernel_instances": custom,
              "custom_kernel_count": len(custom),
              "custom_kernel_total_us": sum(float(r["Duration(us)"]) for r in custom)}
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({"custom_kernel_count": report["custom_kernel_count"],
                      "custom_kernel_total_us": report["custom_kernel_total_us"],
                      "operator_rows": len(selected)}, indent=2))


if __name__ == "__main__":
    main()
