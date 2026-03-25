#!/usr/bin/env python3
"""Auto-regression checker: compares compare_all.py output against KB baselines.

Usage:
    python check_regression.py                  # check _comparisons/summary.txt against baselines
    python check_regression.py --update         # update baselines after a confirmed improvement
    python check_regression.py --strict         # fail on ANY regression, not just major ones

Exit codes:
    0 = no major regressions
    1 = major regression detected (Feat% dropped >2pp or WdErr worsened >5px)
    2 = file not found or parse error
"""
import sys, os, re, argparse

SUMMARY_PATH = os.path.join("_comparisons", "summary.txt")

# Baselines embedded from kb-baselines.md (source of truth)
# Format: {image_name: {metric: value}}
BASELINES = {
    "Ref":   {"Feat%": 90.6, "Miss%": 5.6, "Xtra%": 27.6, "WdErr": 0.04,  "MnDif": 5.22,  "Nodes": 10266},
    "test2": {"Feat%": 97.2, "Miss%": 0.7, "Xtra%": 2.5,  "WdErr": 25.19, "MnDif": 10.36, "Nodes": 20828},
    "test3": {"Feat%": 87.3, "Miss%": 4.8, "Xtra%": 14.4, "WdErr": 12.11, "MnDif": 1.62,  "Nodes": 15786},
    "test4": {"Feat%": 91.6, "Miss%": 2.4, "Xtra%": 2.3,  "WdErr": 2.56,  "MnDif": 13.31, "Nodes": 123296},
    "test5": {"Feat%": 83.6, "Miss%": 3.1, "Xtra%": 6.8,  "WdErr": 6.48,  "MnDif": 10.09, "Nodes": 53930},
}

# Regression thresholds: how much worse before we flag it
THRESHOLDS_MAJOR = {
    "Feat%": -2.0,   # Feature% dropping >2pp is major
    "Miss%": 2.0,    # Miss% increasing >2pp is major
    "Xtra%": 5.0,    # Extra% increasing >5pp is major
    "WdErr": 5.0,    # WdErr increasing >5px is major
    "MnDif": 3.0,    # MnDif increasing >3 is major
}

THRESHOLDS_MINOR = {
    "Feat%": -0.5,
    "Miss%": 0.5,
    "Xtra%": 1.5,
    "WdErr": 2.0,
    "MnDif": 1.0,
}


def parse_summary(path):
    """Parse summary.txt into {image_name: {metric: value}}."""
    results = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Image") or line.startswith("-"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            name = parts[0]
            try:
                results[name] = {
                    "Feat%": float(parts[1]),
                    "Miss%": float(parts[2]),
                    "Xtra%": float(parts[3]),
                    "WdErr": float(parts[4]),
                    "MnDif": float(parts[5]),
                    "Nodes": int(parts[7].replace(",", "")),
                }
            except (ValueError, IndexError):
                continue
    return results


def check_regressions(current, baselines, strict=False):
    """Compare current vs baselines. Returns (majors, minors) lists."""
    thresholds = THRESHOLDS_MINOR if strict else THRESHOLDS_MAJOR
    majors = []
    minors = []

    for img, baseline in baselines.items():
        if img not in current:
            continue
        cur = current[img]
        for metric, base_val in baseline.items():
            if metric == "Nodes":
                continue  # don't flag node count changes as regressions
            cur_val = cur.get(metric)
            if cur_val is None:
                continue

            # For Feat%, higher is better — regression = current < baseline
            # For Miss%, Xtra%, WdErr, MnDif — lower is better — regression = current > baseline
            if metric == "Feat%":
                delta = cur_val - base_val  # negative = regression
                is_major = delta < THRESHOLDS_MAJOR[metric]
                is_minor = delta < THRESHOLDS_MINOR[metric]
            else:
                delta = cur_val - base_val  # positive = regression
                is_major = delta > THRESHOLDS_MAJOR[metric]
                is_minor = delta > THRESHOLDS_MINOR[metric]

            if is_major:
                majors.append((img, metric, base_val, cur_val, delta))
            elif is_minor and not strict:
                minors.append((img, metric, base_val, cur_val, delta))

    return majors, minors


def print_comparison(current, baselines):
    """Print a side-by-side delta table."""
    print(f"\n{'Image':<8} {'Metric':<7} {'Base':>8} {'Now':>8} {'Delta':>8} {'Status'}")
    print("-" * 52)

    for img in ["Ref", "test2", "test3", "test4", "test5"]:
        if img not in current:
            continue
        cur = current[img]
        base = baselines.get(img, {})
        for metric in ["Feat%", "Miss%", "Xtra%", "WdErr", "MnDif"]:
            b = base.get(metric)
            c = cur.get(metric)
            if b is None or c is None:
                continue
            delta = c - b
            # Determine if change is good, bad, or neutral
            if metric == "Feat%":
                status = "✓" if delta >= 0 else ("✗ MAJOR" if delta < THRESHOLDS_MAJOR[metric] else ("· minor" if delta < THRESHOLDS_MINOR[metric] else "~"))
            else:
                status = "✓" if delta <= 0 else ("✗ MAJOR" if delta > THRESHOLDS_MAJOR[metric] else ("· minor" if delta > THRESHOLDS_MINOR[metric] else "~"))

            print(f"{img:<8} {metric:<7} {b:>8.2f} {c:>8.2f} {delta:>+8.2f} {status}")
        print()


def update_baselines_in_script(current):
    """Update the BASELINES dict in this script from current results."""
    script_path = os.path.abspath(__file__)
    with open(script_path, "r") as f:
        content = f.read()

    new_baselines = "BASELINES = {\n"
    for img in ["Ref", "test2", "test3", "test4", "test5"]:
        if img not in current:
            continue
        c = current[img]
        new_baselines += f'    "{img}":' + " " * (5 - len(img))
        new_baselines += f'{{"Feat%": {c["Feat%"]}, "Miss%": {c["Miss%"]}, "Xtra%": {c["Xtra%"]}, '
        new_baselines += f'"WdErr": {c["WdErr"]}, "MnDif": {c["MnDif"]}, "Nodes": {c["Nodes"]}}},\n'
    new_baselines += "}"

    # Replace the BASELINES block
    pattern = r"BASELINES = \{[^}]+\}"
    content_new = re.sub(pattern, new_baselines, content, count=1, flags=re.DOTALL)

    with open(script_path, "w") as f:
        f.write(content_new)
    print("✓ Baselines updated in check_regression.py")
    print("  Remember to also update /memories/repo/kb-baselines.md!")


def main():
    parser = argparse.ArgumentParser(description="Check compare_all.py output for regressions against baselines.")
    parser.add_argument("--update", action="store_true", help="Update baselines from current summary.txt")
    parser.add_argument("--strict", action="store_true", help="Flag minor regressions as major")
    parser.add_argument("--summary", default=SUMMARY_PATH, help="Path to summary.txt")
    args = parser.parse_args()

    if not os.path.exists(args.summary):
        print(f"✗ {args.summary} not found. Run compare_all.py first.")
        sys.exit(2)

    current = parse_summary(args.summary)
    if not current:
        print(f"✗ Could not parse {args.summary}")
        sys.exit(2)

    if args.update:
        update_baselines_in_script(current)
        return

    print_comparison(current, BASELINES)

    majors, minors = check_regressions(current, BASELINES, strict=args.strict)

    if majors:
        print("=" * 52)
        print(f"✗ {len(majors)} MAJOR REGRESSION(S) DETECTED:")
        for img, metric, base, cur, delta in majors:
            print(f"  {img} {metric}: {base:.2f} → {cur:.2f} ({delta:+.2f})")
        print("\nDo NOT accept this change without investigation.")
        if minors:
            print(f"\n(Also {len(minors)} minor regression(s))")
        sys.exit(1)
    elif minors:
        print("=" * 52)
        print(f"~ {len(minors)} minor regression(s) — review but likely acceptable:")
        for img, metric, base, cur, delta in minors:
            print(f"  {img} {metric}: {base:.2f} → {cur:.2f} ({delta:+.2f})")
        print("\nNo MAJOR regressions. Consider if the tradeoff is worth it.")
    else:
        print("=" * 52)
        print("✓ No regressions detected. Change looks safe.")

    # Show improvements
    improvements = []
    for img, base in BASELINES.items():
        if img not in current:
            continue
        cur = current[img]
        for metric in ["Feat%", "Miss%", "Xtra%", "WdErr", "MnDif"]:
            b = base.get(metric)
            c = cur.get(metric)
            if b is None or c is None:
                continue
            delta = c - b
            if metric == "Feat%" and delta > 0.5:
                improvements.append((img, metric, b, c, delta))
            elif metric != "Feat%" and delta < -0.5:
                improvements.append((img, metric, b, c, delta))

    if improvements:
        print(f"\n★ {len(improvements)} improvement(s):")
        for img, metric, base, cur, delta in improvements:
            print(f"  {img} {metric}: {base:.2f} → {cur:.2f} ({delta:+.2f})")


if __name__ == "__main__":
    main()
