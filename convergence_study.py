#!/usr/bin/env python3

import argparse
import csv
import math
import os
import re
import subprocess
import sys


VEL_L2_RE = re.compile(r"^L_2\s+error:\s*([0-9.eE+-]+)", re.MULTILINE)
RHO_L2_RE = re.compile(r"^rho L_2 error:\s*([0-9.eE+-]+)", re.MULTILINE)
ZONES_RE = re.compile(r"Number of zones in the serial mesh:\s*(\d+)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Laghos p=9 manufactured TGV convergence study."
    )
    parser.add_argument("--laghos", default="./laghos")
    parser.add_argument("--mpirun", default="mpirun")
    parser.add_argument("--np", type=int, default=8)
    parser.add_argument("--orders", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--refinements", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--tf", type=float, default=0.5)
    parser.add_argument("--dt", type=float, default=-1.0)
    parser.add_argument("--ode-solver", type=int, default=4)
    parser.add_argument("--cg-tol", type=float, default=1e-10)
    parser.add_argument("--hypervisc", action="store_true")
    parser.add_argument("--hv-coeff", type=float, default=0.25)
    parser.add_argument("--hv-z", type=int, default=1)
    parser.add_argument("--hv-smooth-steps", type=int, default=5)
    parser.add_argument("--hv-smooth-omega", type=float, default=2.0 / 3.0)
    parser.add_argument("--csv", default="convergence_results.csv")
    parser.add_argument("--log-dir", default="convergence_logs")
    return parser.parse_args()


def last_float(text, regex):
    matches = regex.findall(text)
    return float(matches[-1]) if matches else float("nan")


def run_case(args, order, rs):
    ot = order - 1
    mode = "hv" if args.hypervisc else "stdav"
    log_path = os.path.join(args.log_dir, f"{mode}_q{order}_rs{rs}_np{args.np}.log")

    cmd = [
        args.mpirun, "-np", str(args.np), args.laghos,
        "-p", "9",
        "-dim", "2",
        "-rs", str(rs),
        "-ok", str(order),
        "-ot", str(ot),
        "-tf", str(args.tf),
        "-s", str(args.ode_solver),
        "-pa",
        "-iv",
        "-cgt", str(args.cg_tol),
        "-no-cond",
        "-no-vis",
        "-no-visit",
        "-no-print",
    ]

    if args.dt > 0.0:
        cmd += ["-dt", str(args.dt)]

    if args.hypervisc:
        cmd += [
            "--hypervisc",
            "--hv-coeff", str(args.hv_coeff),
            "--hv-z", str(args.hv_z),
            "--hv-smooth-steps", str(args.hv_smooth_steps),
            "--hv-smooth-omega", str(args.hv_smooth_omega),
        ]

    print("Running:", " ".join(cmd), flush=True)
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=os.getcwd(),
    )

    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write(proc.stdout)

    zones_match = ZONES_RE.search(proc.stdout)
    zones = int(zones_match.group(1)) if zones_match else -1
    h = 1.0 / math.sqrt(zones) if zones > 0 else float("nan")

    return {
        "mode": mode,
        "order": order,
        "order_thermo": ot,
        "refinement": rs,
        "np": args.np,
        "zones": zones,
        "h": h,
        "vel_l2_error": last_float(proc.stdout, VEL_L2_RE),
        "rho_l2_error": last_float(proc.stdout, RHO_L2_RE),
        "status": "ok" if proc.returncode == 0 else "failed",
        "log_file": log_path,
    }


def rate(prev_err, curr_err):
    if prev_err <= 0.0 or curr_err <= 0.0:
        return float("nan")
    return math.log(prev_err / curr_err, 2.0)


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    rows = []
    for order in args.orders:
        for rs in args.refinements:
            rows.append(run_case(args, order, rs))

    rows.sort(key=lambda row: (row["order"], row["refinement"]))
    previous = {}
    for row in rows:
        prev = previous.get(row["order"])
        if prev is None:
            row["vel_l2_rate"] = float("nan")
            row["rho_l2_rate"] = float("nan")
        else:
            row["vel_l2_rate"] = rate(prev["vel_l2_error"], row["vel_l2_error"])
            row["rho_l2_rate"] = rate(prev["rho_l2_error"], row["rho_l2_error"])
        previous[row["order"]] = row

    fieldnames = [
        "mode", "order", "order_thermo", "refinement", "np", "zones", "h",
        "vel_l2_error", "vel_l2_rate", "rho_l2_error", "rho_l2_rate",
        "status", "log_file",
    ]

    with open(args.csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.csv}")
    for row in rows:
        print(
            f"{row['mode']} Q{row['order']} rs={row['refinement']} "
            f"vel_L2={row['vel_l2_error']:.6e} rho_L2={row['rho_l2_error']:.6e} "
            f"status={row['status']}"
        )


if __name__ == "__main__":
    sys.exit(main())
