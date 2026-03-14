"""
combine_train.py  –  Runs train.main() for Protocol 2.1 then 2.2.

Usage:
    python combine_train.py                          # uses config.yaml
    python combine_train.py --config config.yaml
"""

import sys
import argparse
import yaml
import train   # reuses all logic from train.py unchanged


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    return p.parse_args()


def run_protocol(cfg_path, protocol_label, train_txt, dev_txt, test_txt,
                 output_dir, cfg):
    """Write a temporary per-protocol config and call train.main()."""
    proto_cfg = {
        "data_root":   cfg["data_root"],
        "train_txt":   train_txt,
        "dev_txt":     dev_txt,
        "test_txt":    test_txt,
        "output_dir":  output_dir,
        # model & training params — same for both
        "clip_model":            cfg.get("clip_model",            "ViT-B/16"),
        "num_student_tokens":    cfg.get("num_student_tokens",    16),
        "num_teacher_templates": cfg.get("num_teacher_templates", 6),
        "lambda_ufm":            cfg.get("lambda_ufm",            1.0),
        "epochs":                cfg.get("epochs",                20),
        "batch_size":            cfg.get("batch_size",            32),
        "accum_steps":           cfg.get("accum_steps",           2),
        "lr":                    cfg.get("lr",                    2e-4),
        "wd":                    cfg.get("wd",                    1e-4),
        "num_workers":           cfg.get("num_workers",           4),
    }

    tmp_cfg_path = f"_tmp_{protocol_label}.yaml"
    with open(tmp_cfg_path, "w") as f:
        yaml.dump(proto_cfg, f)

    print(f"\n{'='*65}")
    print(f"  STARTING  {protocol_label}")
    print(f"{'='*65}")

    # Patch sys.argv so train.parse_args() picks up our temp config
    sys.argv = ["train.py", "--config", tmp_cfg_path]
    train.main()

    import os; os.remove(tmp_cfg_path)


def main(output_dir):
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_protocol(
        cfg_path       = args.config,
        protocol_label = "Protocol 2.1",
        train_txt      = cfg["p21_train_txt"],
        dev_txt        = cfg["p21_dev_txt"],
        test_txt       = cfg["p21_test_txt"],
        output_dir     = output_dir + "_p2.1", 
        cfg            = cfg,
    )

    run_protocol(
        cfg_path       = args.config,
        protocol_label = "Protocol 2.2",
        train_txt      = cfg["p22_train_txt"],
        dev_txt        = cfg["p22_dev_txt"],
        test_txt       = cfg["p22_test_txt"],
        output_dir     = output_dir + "_p2.2",
        cfg            = cfg,
    )

    print("\n" + "="*65)
    print("  DONE — both protocols finished.")
    print(f"  Weights: {output_dir}_p2.1/best_model.pth")
    print(f"           {output_dir}_p2.2/best_model.pth")
    print("="*65)


if __name__ == "__main__":
    output_dir="checkpoints"
    main(output_dir=output_dir)
