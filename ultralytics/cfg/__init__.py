

import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Union

from ultralytics import __version__
from ultralytics.utils import (
    ASSETS,
    DEFAULT_CFG,
    DEFAULT_CFG_DICT,
    DEFAULT_CFG_PATH,
    IS_VSCODE,
    LOGGER,
    RANK,
    ROOT,
    RUNS_DIR,
    SETTINGS,
    SETTINGS_FILE,
    TESTS_RUNNING,
    YAML,
    IterableSimpleNamespace,
    checks,
    colorstr,
    deprecation_warn,
    vscode_msg,
)

SOLUTION_MAP = {
    "count": "ObjectCounter",
    "crop": "ObjectCropper",
    "blur": "ObjectBlurrer",
    "workout": "AIGym",
    "heatmap": "Heatmap",
    "isegment": "InstanceSegmentation",
    "visioneye": "VisionEye",
    "speed": "SpeedEstimator",
    "queue": "QueueManager",
    "analytics": "Analytics",
    "inference": "Inference",
    "trackzone": "TrackZone",
    "help": None,
}

MODES = frozenset({"train", "val", "predict", "export", "track", "benchmark"})
TASKS = frozenset({"detect", "segment", "classify", "pose", "obb"})
TASK2DATA = {
    "detect": "coco8.yaml",
    "segment": "coco8-seg.yaml",
    "classify": "imagenet10",
    "pose": "coco8-pose.yaml",
    "obb": "dota8.yaml",
}
TASK2MODEL = {
    "detect": "yolo11n.pt",
    "segment": "yolo11n-seg.pt",
    "classify": "yolo11n-cls.pt",
    "pose": "yolo11n-pose.pt",
    "obb": "yolo11n-obb.pt",
}
TASK2METRIC = {
    "detect": "metrics/mAP50-95(B)",
    "segment": "metrics/mAP50-95(M)",
    "classify": "metrics/accuracy_top1",
    "pose": "metrics/mAP50-95(P)",
    "obb": "metrics/mAP50-95(B)",
}

ARGV = sys.argv or ["", ""]
SOLUTIONS_HELP_MSG = f
CLI_HELP_MSG = f

CFG_FLOAT_KEYS = frozenset(
    {
        "warmup_epochs",
        "box",
        "cls",
        "dfl",
        "degrees",
        "shear",
        "time",
        "workspace",
        "batch",
    }
)
CFG_FRACTION_KEYS = frozenset(
    {
        "dropout",
        "lr0",
        "lrf",
        "momentum",
        "weight_decay",
        "warmup_momentum",
        "warmup_bias_lr",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "translate",
        "scale",
        "perspective",
        "flipud",
        "fliplr",
        "bgr",
        "mosaic",
        "mixup",
        "cutmix",
        "copy_paste",
        "conf",
        "iou",
        "fraction",
    }
)
CFG_INT_KEYS = frozenset(
    {
        "epochs",
        "patience",
        "workers",
        "seed",
        "close_mosaic",
        "mask_ratio",
        "max_det",
        "vid_stride",
        "line_width",
        "nbs",
        "save_period",
    }
)
CFG_BOOL_KEYS = frozenset(
    {
        "save",
        "exist_ok",
        "verbose",
        "deterministic",
        "single_cls",
        "rect",
        "cos_lr",
        "overlap_mask",
        "val",
        "save_json",
        "half",
        "dnn",
        "plots",
        "show",
        "save_txt",
        "save_conf",
        "save_crop",
        "save_frames",
        "show_labels",
        "show_conf",
        "visualize",
        "augment",
        "agnostic_nms",
        "retina_masks",
        "show_boxes",
        "keras",
        "optimize",
        "int8",
        "dynamic",
        "simplify",
        "nms",
        "profile",
        "multi_scale",
    }
)

def cfg2dict(cfg: Union[str, Path, Dict, SimpleNamespace]) -> Dict:

    if isinstance(cfg, (str, Path)):
        cfg = YAML.load(cfg)
    elif isinstance(cfg, SimpleNamespace):
        cfg = vars(cfg)
    return cfg

def get_cfg(cfg: Union[str, Path, Dict, SimpleNamespace] = DEFAULT_CFG_DICT, overrides: Dict = None) -> SimpleNamespace:

    cfg = cfg2dict(cfg)

    if overrides:
        overrides = cfg2dict(overrides)
        if "save_dir" not in cfg:
            overrides.pop("save_dir", None)
        check_dict_alignment(cfg, overrides)
        cfg = {**cfg, **overrides}

    for k in "project", "name":
        if k in cfg and isinstance(cfg[k], (int, float)):
            cfg[k] = str(cfg[k])
    if cfg.get("name") == "model":
        cfg["name"] = str(cfg.get("model", "")).partition(".")[0]
        LOGGER.warning(f"'name=model' automatically updated to 'name={cfg['name']}'.")

    check_cfg(cfg)

    return IterableSimpleNamespace(**cfg)

def check_cfg(cfg: Dict, hard: bool = True) -> None:

    for k, v in cfg.items():
        if v is not None:
            if k in CFG_FLOAT_KEYS and not isinstance(v, (int, float)):
                if hard:
                    raise TypeError(
                        f"'{k}={v}' is of invalid type {type(v).__name__}. "
                        f"Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')"
                    )
                cfg[k] = float(v)
            elif k in CFG_FRACTION_KEYS:
                if not isinstance(v, (int, float)):
                    if hard:
                        raise TypeError(
                            f"'{k}={v}' is of invalid type {type(v).__name__}. "
                            f"Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')"
                        )
                    cfg[k] = v = float(v)
                if not (0.0 <= v <= 1.0):
                    raise ValueError(f"'{k}={v}' is an invalid value. Valid '{k}' values are between 0.0 and 1.0.")
            elif k in CFG_INT_KEYS and not isinstance(v, int):
                if hard:
                    raise TypeError(
                        f"'{k}={v}' is of invalid type {type(v).__name__}. '{k}' must be an int (i.e. '{k}=8')"
                    )
                cfg[k] = int(v)
            elif k in CFG_BOOL_KEYS and not isinstance(v, bool):
                if hard:
                    raise TypeError(
                        f"'{k}={v}' is of invalid type {type(v).__name__}. "
                        f"'{k}' must be a bool (i.e. '{k}=True' or '{k}=False')"
                    )
                cfg[k] = bool(v)

def get_save_dir(args: SimpleNamespace, name: str = None) -> Path:

    if getattr(args, "save_dir", None):
        save_dir = args.save_dir
    else:
        from ultralytics.utils.files import increment_path

        project = args.project or (ROOT.parent / "tests/tmp/runs" if TESTS_RUNNING else RUNS_DIR) / args.task
        name = name or args.name or f"{args.mode}"
        save_dir = increment_path(Path(project) / name, exist_ok=args.exist_ok if RANK in {-1, 0} else True)

    return Path(save_dir)

def _handle_deprecation(custom: Dict) -> Dict:

    deprecated_mappings = {
        "boxes": ("show_boxes", lambda v: v),
        "hide_labels": ("show_labels", lambda v: not bool(v)),
        "hide_conf": ("show_conf", lambda v: not bool(v)),
        "line_thickness": ("line_width", lambda v: v),
    }
    removed_keys = {"label_smoothing", "save_hybrid", "crop_fraction"}

    for old_key, (new_key, transform) in deprecated_mappings.items():
        if old_key not in custom:
            continue
        deprecation_warn(old_key, new_key)
        custom[new_key] = transform(custom.pop(old_key))

    for key in removed_keys:
        if key not in custom:
            continue
        deprecation_warn(key)
        custom.pop(key)

    return custom

def check_dict_alignment(base: Dict, custom: Dict, e: Exception = None) -> None:

    custom = _handle_deprecation(custom)
    base_keys, custom_keys = (frozenset(x.keys()) for x in (base, custom))
    if mismatched := [k for k in custom_keys if k not in base_keys]:
        from difflib import get_close_matches

        string = ""
        for x in mismatched:
            matches = get_close_matches(x, base_keys)
            matches = [f"{k}={base[k]}" if base.get(k) is not None else k for k in matches]
            match_str = f"Similar arguments are i.e. {matches}." if matches else ""
            string += f"'{colorstr('red', 'bold', x)}' is not a valid YOLO argument. {match_str}\n"
        raise SyntaxError(string + CLI_HELP_MSG) from e

def merge_equals_args(args: List[str]) -> List[str]:

    new_args = []
    current = ""
    depth = 0

    i = 0
    while i < len(args):
        arg = args[i]

        if arg == "=" and 0 < i < len(args) - 1:
            new_args[-1] += f"={args[i + 1]}"
            i += 2
            continue
        elif arg.endswith("=") and i < len(args) - 1 and "=" not in args[i + 1]:
            new_args.append(f"{arg}{args[i + 1]}")
            i += 2
            continue
        elif arg.startswith("=") and i > 0:
            new_args[-1] += arg
            i += 1
            continue

        depth += arg.count("[") - arg.count("]")
        current += arg
        if depth == 0:
            new_args.append(current)
            current = ""

        i += 1

    if current:
        new_args.append(current)

    return new_args

def handle_yolo_hub(args: List[str]) -> None:

    from ultralytics import hub

    if args[0] == "login":
        key = args[1] if len(args) > 1 else ""

        hub.login(key)
    elif args[0] == "logout":

        hub.logout()

def handle_yolo_settings(args: List[str]) -> None:

    url = "https://docs.ultralytics.com/quickstart/#ultralytics-settings"
    try:
        if any(args):
            if args[0] == "reset":
                SETTINGS_FILE.unlink()
                SETTINGS.reset()
                LOGGER.info("Settings reset successfully")
            else:
                new = dict(parse_key_value_pair(a) for a in args)
                check_dict_alignment(SETTINGS, new)
                SETTINGS.update(new)
                for k, v in new.items():
                    LOGGER.info(f"✅ Updated '{k}={v}'")

        LOGGER.info(SETTINGS)
        LOGGER.info(f"💡 Learn more about Ultralytics Settings at {url}")
    except Exception as e:
        LOGGER.warning(f"settings error: '{e}'. Please see {url} for help.")

def handle_yolo_solutions(args: List[str]) -> None:

    from ultralytics.solutions.config import SolutionConfig

    full_args_dict = vars(SolutionConfig())
    overrides = {}

    for arg in merge_equals_args(args):
        arg = arg.lstrip("-").rstrip(",")
        if "=" in arg:
            try:
                k, v = parse_key_value_pair(arg)
                overrides[k] = v
            except (NameError, SyntaxError, ValueError, AssertionError) as e:
                check_dict_alignment(full_args_dict, {arg: ""}, e)
        elif arg in full_args_dict and isinstance(full_args_dict.get(arg), bool):
            overrides[arg] = True
    check_dict_alignment(full_args_dict, overrides)

    if not args:
        LOGGER.warning("No solution name provided. i.e `yolo solutions count`. Defaulting to 'count'.")
        args = ["count"]
    if args[0] == "help":
        LOGGER.info(SOLUTIONS_HELP_MSG)
        return
    elif args[0] in SOLUTION_MAP:
        solution_name = args.pop(0)
    else:
        LOGGER.warning(
            f"❌ '{args[0]}' is not a valid solution. 💡 Defaulting to 'count'.\n"
            f"🚀 Available solutions: {', '.join(list(SOLUTION_MAP.keys())[:-1])}\n"
        )
        solution_name = "count"

    if solution_name == "inference":
        checks.check_requirements("streamlit>=1.29.0")
        LOGGER.info("💡 Loading Ultralytics live inference app...")
        subprocess.run(
            [
                "streamlit",
                "run",
                str(ROOT / "solutions/streamlit_inference.py"),
                "--server.headless",
                "true",
                overrides.pop("model", "yolo11n.pt"),
            ]
        )
    else:
        import cv2

        from ultralytics import solutions

        solution = getattr(solutions, SOLUTION_MAP[solution_name])(is_cli=True, **overrides)

        cap = cv2.VideoCapture(solution.CFG["source"])
        if solution_name != "crop":

            w, h, fps = (
                int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
            )
            if solution_name == "analytics":
                w, h = 1280, 720
            save_dir = get_save_dir(SimpleNamespace(project="runs/solutions", name="exp", exist_ok=False))
            save_dir.mkdir(parents=True)
            vw = cv2.VideoWriter(str(save_dir / f"{solution_name}.avi"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        try:
            f_n = 0
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                results = solution(frame, f_n := f_n + 1) if solution_name == "analytics" else solution(frame)
                if solution_name != "crop":
                    vw.write(results.plot_im)
                if solution.CFG["show"] and cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()

def parse_key_value_pair(pair: str = "key=value") -> tuple:

    k, v = pair.split("=", 1)
    k, v = k.strip(), v.strip()
    assert v, f"missing '{k}' value"
    return k, smart_value(v)

def smart_value(v: str) -> Any:

    v_lower = v.lower()
    if v_lower == "none":
        return None
    elif v_lower == "true":
        return True
    elif v_lower == "false":
        return False
    else:
        try:
            return eval(v)
        except Exception:
            return v

def entrypoint(debug: str = "") -> None:

    args = (debug.split(" ") if debug else ARGV)[1:]
    if not args:
        LOGGER.info(CLI_HELP_MSG)
        return

    special = {
        "help": lambda: LOGGER.info(CLI_HELP_MSG),
        "checks": checks.collect_system_info,
        "version": lambda: LOGGER.info(__version__),
        "settings": lambda: handle_yolo_settings(args[1:]),
        "cfg": lambda: YAML.print(DEFAULT_CFG_PATH),
        "hub": lambda: handle_yolo_hub(args[1:]),
        "login": lambda: handle_yolo_hub(args),
        "logout": lambda: handle_yolo_hub(args),
        "copy-cfg": copy_default_cfg,
        "solutions": lambda: handle_yolo_solutions(args[1:]),
    }
    full_args_dict = {**DEFAULT_CFG_DICT, **{k: None for k in TASKS}, **{k: None for k in MODES}, **special}

    special.update({k[0]: v for k, v in special.items()})
    special.update({k[:-1]: v for k, v in special.items() if len(k) > 1 and k.endswith("s")})
    special = {**special, **{f"-{k}": v for k, v in special.items()}, **{f"--{k}": v for k, v in special.items()}}

    overrides = {}
    for a in merge_equals_args(args):
        if a.startswith("--"):
            LOGGER.warning(f"argument '{a}' does not require leading dashes '--', updating to '{a[2:]}'.")
            a = a[2:]
        if a.endswith(","):
            LOGGER.warning(f"argument '{a}' does not require trailing comma ',', updating to '{a[:-1]}'.")
            a = a[:-1]
        if "=" in a:
            try:
                k, v = parse_key_value_pair(a)
                if k == "cfg" and v is not None:
                    LOGGER.info(f"Overriding {DEFAULT_CFG_PATH} with {v}")
                    overrides = {k: val for k, val in YAML.load(checks.check_yaml(v)).items() if k != "cfg"}
                else:
                    overrides[k] = v
            except (NameError, SyntaxError, ValueError, AssertionError) as e:
                check_dict_alignment(full_args_dict, {a: ""}, e)

        elif a in TASKS:
            overrides["task"] = a
        elif a in MODES:
            overrides["mode"] = a
        elif a.lower() in special:
            special[a.lower()]()
            return
        elif a in DEFAULT_CFG_DICT and isinstance(DEFAULT_CFG_DICT[a], bool):
            overrides[a] = True
        elif a in DEFAULT_CFG_DICT:
            raise SyntaxError(
                f"'{colorstr('red', 'bold', a)}' is a valid YOLO argument but is missing an '=' sign "
                f"to set its value, i.e. try '{a}={DEFAULT_CFG_DICT[a]}'\n{CLI_HELP_MSG}"
            )
        else:
            check_dict_alignment(full_args_dict, {a: ""})

    check_dict_alignment(full_args_dict, overrides)

    mode = overrides.get("mode")
    if mode is None:
        mode = DEFAULT_CFG.mode or "predict"
        LOGGER.warning(f"'mode' argument is missing. Valid modes are {list(MODES)}. Using default 'mode={mode}'.")
    elif mode not in MODES:
        raise ValueError(f"Invalid 'mode={mode}'. Valid modes are {list(MODES)}.\n{CLI_HELP_MSG}")

    task = overrides.pop("task", None)
    if task:
        if task not in TASKS:
            if task == "track":
                LOGGER.warning(
                    f"invalid 'task=track', setting 'task=detect' and 'mode=track'. Valid tasks are {list(TASKS)}.\n{CLI_HELP_MSG}."
                )
                task, mode = "detect", "track"
            else:
                raise ValueError(f"Invalid 'task={task}'. Valid tasks are {list(TASKS)}.\n{CLI_HELP_MSG}")
        if "model" not in overrides:
            overrides["model"] = TASK2MODEL[task]

    model = overrides.pop("model", DEFAULT_CFG.model)
    if model is None:
        model = "yolo11n.pt"
        LOGGER.warning(f"'model' argument is missing. Using default 'model={model}'.")
    overrides["model"] = model
    stem = Path(model).stem.lower()
    if "rtdetr" in stem:
        from ultralytics import RTDETR

        model = RTDETR(model)
    elif "fastsam" in stem:
        from ultralytics import FastSAM

        model = FastSAM(model)
    elif "sam_" in stem or "sam2_" in stem or "sam2.1_" in stem:
        from ultralytics import SAM

        model = SAM(model)
    else:
        from ultralytics import YOLO

        model = YOLO(model, task=task)
        if "yoloe" in stem or "world" in stem:
            cls_list = overrides.pop("classes", DEFAULT_CFG.classes)
            if cls_list is not None and isinstance(cls_list, str):
                model.set_classes(cls_list.split(","))

    if task != model.task:
        if task:
            LOGGER.warning(
                f"conflicting 'task={task}' passed with 'task={model.task}' model. "
                f"Ignoring 'task={task}' and updating to 'task={model.task}' to match model."
            )
        task = model.task

    if mode in {"predict", "track"} and "source" not in overrides:
        overrides["source"] = (
            "https://ultralytics.com/images/boats.jpg" if task == "obb" else DEFAULT_CFG.source or ASSETS
        )
        LOGGER.warning(f"'source' argument is missing. Using default 'source={overrides['source']}'.")
    elif mode in {"train", "val"}:
        if "data" not in overrides and "resume" not in overrides:
            overrides["data"] = DEFAULT_CFG.data or TASK2DATA.get(task or DEFAULT_CFG.task, DEFAULT_CFG.data)
            LOGGER.warning(f"'data' argument is missing. Using default 'data={overrides['data']}'.")
    elif mode == "export":
        if "format" not in overrides:
            overrides["format"] = DEFAULT_CFG.format or "torchscript"
            LOGGER.warning(f"'format' argument is missing. Using default 'format={overrides['format']}'.")

    getattr(model, mode)(**overrides)

    LOGGER.info(f"💡 Learn more at https://docs.ultralytics.com/modes/{mode}")

    if IS_VSCODE and SETTINGS.get("vscode_msg", True):
        LOGGER.info(vscode_msg())

def copy_default_cfg() -> None:

    new_file = Path.cwd() / DEFAULT_CFG_PATH.name.replace(".yaml", "_copy.yaml")
    shutil.copy2(DEFAULT_CFG_PATH, new_file)
    LOGGER.info(
        f"{DEFAULT_CFG_PATH} copied to {new_file}\n"
        f"Example YOLO command with this new custom cfg:\n    yolo cfg='{new_file}' imgsz=320 batch=8"
    )

if __name__ == "__main__":

    entrypoint(debug="")
