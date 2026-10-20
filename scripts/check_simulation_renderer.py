"""Probe the actual MuJoCo OpenGL renderer without running a policy episode.

Run once per simulator interpreter on the evaluation Pod, for example:

  .venv-eval-aloha/bin/python scripts/check_simulation_renderer.py --dataset aloha --mujoco-gl egl
  .venv-eval-libero/bin/python scripts/check_simulation_renderer.py --dataset libero --mujoco-gl egl

The probe initializes and resets one environment, renders one small frame, then
closes it. It never calls ``env.step`` or evaluates a checkpoint. The report
keeps CUDA inference, OpenGL rendering, and MuJoCo physics separate: standard
MuJoCo here advances physics on CPU; EGL/GL only accelerates image rendering.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import glob
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _egl_devices() -> dict:
    """Enumerate EGL devices using the system libEGL, without creating a context."""
    library = ctypes.util.find_library("EGL")
    if not library:
        return {"available": False, "reason": "libEGL not found", "devices": []}
    try:
        egl = ctypes.CDLL(library)
        get_proc = egl.eglGetProcAddress
        get_proc.argtypes = [ctypes.c_char_p]
        get_proc.restype = ctypes.c_void_p
        query_ptr = get_proc(b"eglQueryDevicesEXT")
        query_string_ptr = get_proc(b"eglQueryDeviceStringEXT")
        if not query_ptr or not query_string_ptr:
            return {"available": False, "reason": "EGL_EXT_device_enumeration/query unavailable",
                    "library": library, "devices": []}

        query = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_int,
                                 ctypes.POINTER(ctypes.c_void_p),
                                 ctypes.POINTER(ctypes.c_int))(query_ptr)
        query_string = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p,
                                        ctypes.c_int)(query_string_ptr)
        query_attr_ptr = get_proc(b"eglQueryDeviceAttribEXT")
        query_attr = (ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_void_p, ctypes.c_int,
                                       ctypes.POINTER(ctypes.c_ssize_t))(query_attr_ptr)
                      if query_attr_ptr else None)
        count = ctypes.c_int(0)
        if not query(0, None, ctypes.byref(count)):
            return {"available": False, "reason": "eglQueryDevicesEXT count query failed",
                    "library": library, "devices": []}
        handles = (ctypes.c_void_p * count.value)()
        found = ctypes.c_int(0)
        if count.value and not query(count.value, handles, ctypes.byref(found)):
            return {"available": False, "reason": "eglQueryDevicesEXT enumeration failed",
                    "library": library, "devices": []}

        def decode(handle, name: int):
            value = query_string(handle, name)
            return value.decode("utf-8", errors="replace") if value else None

        devices = []
        # EGL_EXT_device_query constants.
        for index in range(found.value):
            handle = handles[index]
            extensions = decode(handle, 0x3055) or ""
            item = {"index": index, "vendor": decode(handle, 0x3053),
                    "extensions": extensions.split()}
            if "EGL_NV_device_cuda" in extensions and query_attr is not None:
                cuda_device = ctypes.c_ssize_t(-1)
                if query_attr(handle, 0x323A, ctypes.byref(cuda_device)):
                    item["cuda_device_index"] = cuda_device.value
            if "EGL_EXT_device_drm" in extensions:
                item["drm_device_file"] = decode(handle, 0x3233)
            if "EGL_EXT_device_drm_render_node" in extensions:
                item["drm_render_node_file"] = decode(handle, 0x3377)
            devices.append(item)
        return {"available": True, "library": library, "count": found.value,
                "devices": devices}
    except Exception as exc:  # diagnostic should report missing optional EGL APIs
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}", "devices": []}


def _host_devices() -> dict:
    nvidia = {"available": False, "output": None}
    executable = shutil.which("nvidia-smi")
    if executable:
        try:
            completed = subprocess.run(
                [executable, "--query-gpu=index,name,uuid", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10, check=False)
            nvidia = {"available": completed.returncode == 0,
                      "returncode": completed.returncode,
                      "output": completed.stdout.strip() or completed.stderr.strip()}
        except Exception as exc:
            nvidia["error"] = f"{type(exc).__name__}: {exc}"
    return {"nvidia_smi": nvidia,
            "dri_render_nodes": sorted(glob.glob("/dev/dri/renderD*")),
            "egl": _egl_devices()}


def _cuda_report() -> dict:
    try:
        import torch
        available = bool(torch.cuda.is_available())
        return {"torch_version": torch.__version__, "available": available,
                "device_count": torch.cuda.device_count() if available else 0,
                "devices": ([torch.cuda.get_device_name(index)
                             for index in range(torch.cuda.device_count())]
                            if available else []),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    except Exception as exc:
        return {"available": None, "error": f"{type(exc).__name__}: {exc}"}


def _gl_report() -> dict:
    try:
        from OpenGL import GL
        def value(name):
            raw = GL.glGetString(getattr(GL, name))
            return raw.decode("utf-8", errors="replace") if raw else None
        return {"vendor": value("GL_VENDOR"), "renderer": value("GL_RENDERER"),
                "version": value("GL_VERSION")}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _renderer_class(gl: dict) -> str:
    renderer = str(gl.get("renderer") or "").lower()
    vendor = str(gl.get("vendor") or "").lower()
    if any(token in renderer for token in ("llvmpipe", "softpipe", "swrast", "software rasterizer")):
        return "software"
    if "nvidia" in vendor or "nvidia" in renderer:
        return "nvidia_hardware"
    if any(token in f"{vendor} {renderer}" for token in
           ("amd", "radeon", "intel", "arc graphics")):
        return "hardware_non_nvidia"
    return "unresolved"


def _probe_aloha() -> tuple[dict, object | None]:
    from configs.registry import get_config
    import gymnasium as gym
    import gym_aloha  # noqa: F401 - registers the environment

    cfg = get_config("aloha", "exp05")
    env = gym.make(cfg.env_id, obs_type="pixels_agent_pos",
                   max_episode_steps=cfg.sim_max_steps, disable_env_checker=True)
    try:
        observation, _ = env.reset(seed=12345)
        physics = env.unwrapped._env.physics
        # The reset already requests the policy image; this small explicit draw
        # ensures the OpenGL context is current when GL_VENDOR/RENDERER are read.
        frame = physics.render(height=32, width=32, camera_id=0)
        # dm_control owns a render executor; query on its GL thread while that
        # EGL/GLFW context is explicitly current, rather than trusting whatever
        # context happens to be current on Python's main thread.
        with physics.contexts.gl.make_current() as context:
            gl = context.call(_gl_report)
        result = {"dataset": "aloha", "experiment_config": "exp05",
                  "environment_id": cfg.env_id, "engine": "MuJoCo via dm_control",
                  "render_backend": os.environ.get("MUJOCO_GL"),
                  "egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
                  "display": os.environ.get("DISPLAY"),
                  "reset_observation_keys": sorted(observation),
                  "probe_frame_shape": list(frame.shape), "gl": gl,
                  "renderer_class": _renderer_class(gl),
                  "physics_backend": "standard MuJoCo CPU integration",
                  "task_steps_executed": 0}
        return result, env
    except Exception:
        env.close()
        raise


def _probe_libero() -> tuple[dict, object | None]:
    from configs.registry import get_config
    from envs.libero_env import make_env
    from libero.libero import benchmark, get_libero_path

    cfg = get_config("libero", "exp05")
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    task = suite.get_task(0)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = make_env(bddl, cfg)
    try:
        observation = env.reset()
        frame = env.sim.render(width=32, height=32, camera_name="agentview")
        gl = _gl_report()
        result = {"dataset": "libero", "experiment_config": "exp05",
                  "benchmark": "libero_goal", "probe_task_index": 0,
                  "engine": "MuJoCo via robosuite/LIBERO OffScreenRenderEnv",
                  "render_backend": os.environ.get("MUJOCO_GL"),
                  "egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
                  "display": os.environ.get("DISPLAY"),
                  "reset_observation_keys": sorted(observation),
                  "probe_frame_shape": list(frame.shape), "gl": gl,
                  "renderer_class": _renderer_class(gl),
                  "physics_backend": "standard MuJoCo CPU integration",
                  "task_steps_executed": 0}
        return result, env
    except Exception:
        env.close()
        raise


def run_probe(dataset: str, backend: str) -> dict:
    os.environ["MUJOCO_GL"] = backend
    if dataset == "aloha":
        report, env = _probe_aloha()
    else:
        report, env = _probe_libero()
    try:
        report["policy_cuda"] = _cuda_report()
        report["gpu_and_gl_devices"] = _host_devices()
        report["interpretation"] = (
            "A CUDA policy device does not prove that the simulator renderer uses that GPU. "
            "Check GL_VENDOR/GL_RENDERER; llvmpipe/softpipe/swrast means software rendering. "
            "MuJoCo physics remains CPU-based in this benchmark; EGL accelerates rendering only.")
        return report
    finally:
        if env is not None:
            env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("aloha", "libero", "both"), required=True)
    parser.add_argument("--mujoco-gl", choices=("egl", "glfw", "osmesa"), default="egl",
                        help="Backend to probe. EGL is recommended for headless GPU rendering.")
    parser.add_argument("--aloha-python", default=sys.executable,
                        help="Interpreter containing gym-aloha/dm_control")
    parser.add_argument("--libero-python", default=sys.executable,
                        help="Interpreter containing the pinned LIBERO/robosuite environment")
    parser.add_argument("--require-hardware-renderer", action="store_true",
                        help="Fail closed unless GL identifies a known GPU renderer")
    args = parser.parse_args(argv)

    if args.dataset == "both":
        overall = 0
        for dataset, interpreter in (("aloha", args.aloha_python), ("libero", args.libero_python)):
            command = [interpreter, str(Path(__file__).resolve()), "--dataset", dataset,
                       "--mujoco-gl", args.mujoco_gl]
            if args.require_hardware_renderer:
                command.append("--require-hardware-renderer")
            env = dict(os.environ, MUJOCO_GL=args.mujoco_gl, PYTHONUNBUFFERED="1")
            if dataset == "aloha" and args.mujoco_gl == "glfw" and not env.get("DISPLAY"):
                xvfb = shutil.which("xvfb-run")
                if not xvfb:
                    print(json.dumps({"dataset": dataset, "status": "failed",
                                      "error": "GLFW probe requires DISPLAY or xvfb-run"}), flush=True)
                    overall = 1
                    continue
                command = [xvfb, "-a", *command]
            completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
            overall = max(overall, completed.returncode)
        return overall

    os.environ["MUJOCO_GL"] = args.mujoco_gl
    if args.dataset == "aloha" and args.mujoco_gl == "glfw" and not os.environ.get("DISPLAY"):
        xvfb = shutil.which("xvfb-run")
        if not xvfb:
            parser.error("GLFW probe requires DISPLAY or xvfb-run; try --mujoco-gl egl")
        command = [xvfb, "-a", sys.executable, str(Path(__file__).resolve()),
                   "--dataset", "aloha", "--mujoco-gl", "glfw"]
        return subprocess.run(command, cwd=ROOT, env=dict(os.environ), check=False).returncode
    try:
        report = run_probe(args.dataset, args.mujoco_gl)
        report["status"] = "ok" if report["gl"].get("renderer") else "renderer_unresolved"
        print(json.dumps(report, sort_keys=True), flush=True)
        if report["status"] != "ok":
            return 2
        if report["renderer_class"] == "software":
            return 3
        if args.require_hardware_renderer and report["renderer_class"] not in {
                "nvidia_hardware", "hardware_non_nvidia"}:
            return 4
        return 0
    except Exception as exc:
        print(json.dumps({"dataset": args.dataset, "render_backend": args.mujoco_gl,
                          "status": "failed", "error_type": type(exc).__name__,
                          "error": str(exc), "traceback": traceback.format_exc()},
                         sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
