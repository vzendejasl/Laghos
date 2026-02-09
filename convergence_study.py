# Usage:
# To run the full convergence study:
# python3 convergence_study.py
#
# To run a single refinement point (e.g., ok=4, ot=3, rp=3):
# python3 convergence_study.py --ok 4 --ot 3 --rp 3

import numpy as np
import subprocess
import csv
import os
import argparse

def run_laghos(rp, ok, ot):
    cmd = [
        "mpirun", "-np", "8", "./laghos",
        "-p", "0", "-dim", "3", "-rs", "1", "-rp", str(rp),
        "-ok", str(ok), "-ot", str(ot),
        "-s", "2", "-tf", ".1", "-fv", "-Re", "400",
        "-cond", "-pr", "0.71", "-ms", "1",
        "-iv", "-diag", "output.txt", "-mach", "0.28",
        "-u0", "1.0", "-cfl", "0.2"
    ]
    print(f"Executing: {' '.join(cmd)}")
    # Use subprocess.Popen to stream output in real-time
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
        for line in proc.stdout:
            print(line, end='')
    if proc.returncode != 0:
        print(f"Error: Laghos exited with code {proc.returncode}")

def get_rms_heat():
    if not os.path.exists("laghos_thermo.csv"):
        return None
    with open("laghos_thermo.csv", "r") as f:
        reader = csv.reader(f)
        lines = list(reader)
        if len(lines) < 2: return None
        try:
            return float(lines[-1][6].strip())
        except: return None

# Constants for Analytical Solution
rho0 = 1.0
u0 = 1.0
gamma = 5.0/3.0
Pr = 0.71
Re = 400.0
mu = 1.0 / (Re * 2.0 * np.pi)
kappa_e = mu * gamma / Pr
rms_analytic = (kappa_e / (rho0**2 * (gamma - 1.0))) * np.pi**2 * rho0 * u0**2 * np.sqrt(6.0)

def main():
    parser = argparse.ArgumentParser(description='Run Laghos convergence study.')
    parser.add_argument('--ok', type=int, help='Kinematic order')
    parser.add_argument('--ot', type=int, help='Thermodynamic order')
    parser.add_argument('--rp', type=int, help='Refinement level')
    args = parser.parse_args()

    results_file = "convergence_results.txt"

    if args.ok is not None and args.ot is not None and args.rp is not None:
        # Run a single refinement point
        if os.path.exists("laghos_thermo.csv"): os.remove("laghos_thermo.csv")
        run_laghos(args.rp, args.ok, args.ot)
        rms = get_rms_heat()
        if rms is None:
            print(f"FAILED: ok={args.ok}, ot={args.ot}, rp={args.rp}")
            return
        
        h = 1.0 / (2**(1 + args.rp))
        err = abs(rms - rms_analytic)
        
        line = f"ok={args.ok}, ot={args.ot}, rp={args.rp}, h={h:.4e}, RMS={rms:.8e}, Error={err:.8e}"
        print(line)
        with open(results_file, "a") as f:
            f.write(line + "\n")
    else:
        # Full study loop
        orders = [(2, 1), (3, 2), (4, 3)]
        refinements = range(5)

        with open(results_file, "w") as f:
            f.write(f"Analytical RMS: {rms_analytic:.8e}\n\n")

        print(f"Analytical RMS: {rms_analytic:.8e}\n")

        for ok, ot in orders:
            header = f"--- Study for Order (ok={ok}, ot={ot}) ---"
            table_header = f"{'rp':<5} | {'h':<10} | {'RMS':<15} | {'Error':<15} | {'Rate':<10}"
            separator = "-" * 65
            
            print(header)
            print(table_header)
            print(separator)
            
            with open(results_file, "a") as f:
                f.write(header + "\n")
                f.write(table_header + "\n")
                f.write(separator + "\n")

            errors = []
            
            # Limit refinement level for 4th order (ok=4) to 4 (rp=0, 1, 2, 3)
            current_refinements = refinements
            if ok == 4:
                current_refinements = range(4)

            for rp in current_refinements:
                if os.path.exists("laghos_thermo.csv"): os.remove("laghos_thermo.csv")
                run_laghos(rp, ok, ot)
                rms = get_rms_heat()
                
                if rms is None:
                    line = f"{rp:<5} | FAILED"
                    print(line)
                    with open(results_file, "a") as f:
                        f.write(line + "\n")
                    continue

                h = 1.0 / (2**(1 + rp))
                err = abs(rms - rms_analytic)
                
                rate_str = "---"
                if len(errors) > 0:
                    rate = np.log2(errors[-1] / err)
                    rate_str = f"{rate:.2f}"
                    
                line = f"{rp:<5} | {h:<10.4e} | {rms:<15.8e} | {err:<15.8e} | {rate_str:<10}"
                print(line)
                with open(results_file, "a") as f:
                    f.write(line + "\n")
                
                errors.append(err)
            print()
            with open(results_file, "a") as f:
                f.write("\n")

if __name__ == "__main__":
    main()