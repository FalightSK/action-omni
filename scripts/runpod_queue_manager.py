"""Demand-driven RunPod evaluator manager (Python standard library only).

The manager never creates or terminates Pods. It may start the named evaluator
when verified queue work is pending and stop it after an empty-queue grace
period. Trainer auto-stop is separately opt-in and gated by durable completion
evidence, no owned training processes, and complete checkpoint/task snapshots.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/runpod_pods.json"
DEFAULT_STATE = ROOT / ".runpod-queue-manager-state.json"
DEFAULT_LOG = ROOT / ".runpod-queue-manager.log.jsonl"
CAP = 60


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def emit(event: str, **fields: Any) -> dict:
    row = {"utc": utc_now(), "event": event, **fields}
    print(json.dumps(row, sort_keys=True, separators=(",", ":")), flush=True)
    return row


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def append_log(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def decrypt_dpapi_secret() -> str:
    """Read a ConvertFrom-SecureString DPAPI value without logging its plaintext."""
    if os.name != "nt":
        raise RuntimeError("RUNPOD_API_KEY is unset and DPAPI secret reading requires Windows")
    local = os.environ.get("LOCALAPPDATA")
    if not local:
        raise RuntimeError("LOCALAPPDATA is unavailable for the RunPod DPAPI secret")
    secret_path = Path(local) / "ActionOmni" / "runpod-api-key.dpapi"
    if not secret_path.is_file():
        raise RuntimeError("RunPod API key is not available in environment or configured DPAPI file")
    ps = r"""
$ErrorActionPreference = 'Stop'
$s = Get-Content -LiteralPath $env:ACTIONOMNI_DPAPI_FILE -Raw | ConvertTo-SecureString
$b = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($s)
try { [Runtime.InteropServices.Marshal]::PtrToStringBSTR($b) }
finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($b) }
"""
    # Prefer PowerShell 7: this credential was provisioned with the local pwsh
    # DPAPI tooling, and Windows PowerShell 5 can fail to import that format.
    shell = shutil.which("pwsh.exe") or shutil.which("pwsh") or shutil.which("powershell.exe")
    if not shell:
        raise RuntimeError("PowerShell is unavailable for the RunPod DPAPI secret")
    result = subprocess.run(
        [shell, "-NoLogo", "-NoProfile", "-NonInteractive", "-ExecutionPolicy",
         "Bypass", "-Command", ps],
                            capture_output=True, text=True, timeout=15, check=False,
                            env={**os.environ, "ACTIONOMNI_DPAPI_FILE": str(secret_path)})
    if result.returncode:
        # Never propagate PowerShell stdout/stderr: either could include sensitive values.
        raise RuntimeError("Could not decrypt the local RunPod DPAPI credential")
    token = result.stdout.strip()
    if not token or "\n" in token:
        raise RuntimeError("Decrypted RunPod credential is empty or malformed")
    return token


def api_token() -> str:
    token = os.environ.get("RUNPOD_API_KEY")
    if token:
        return token.strip()
    return decrypt_dpapi_secret()


def request_api(config: dict, token: str, path: str, method: str = "GET") -> Any:
    url = config["api_base_url"].rstrip("/") + path
    req = urllib.request.Request(url, method=method,
                                 headers={"Authorization": f"Bearer {token}",
                                          "Accept": "application/json",
                                          "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=25) as response:
            body = response.read(4_000_000)
            if response.status < 200 or response.status >= 300:
                raise RuntimeError(f"RunPod API returned HTTP {response.status}")
            if not body:
                return {}
            return json.loads(body.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # Do not include response body; it may contain account data or credentials.
        raise RuntimeError(f"RunPod API returned HTTP {exc.code} for {method} {path}") from None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"RunPod API request failed for {method} {path}: {type(exc).__name__}") from None


def _unwrap_pod(payload: Any) -> dict:
    if isinstance(payload, dict):
        if isinstance(payload.get("pod"), dict):
            return payload["pod"]
        if "content" in payload and isinstance(payload["content"], list):
            for block in payload["content"]:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    try:
                        return _unwrap_pod(json.loads(block["text"]))
                    except json.JSONDecodeError:
                        continue
        return payload
    if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], dict):
        return _unwrap_pod(payload[0])
    raise ValueError("Unrecognized RunPod pod response")


def pod_status(pod: dict, expected_id: str) -> str:
    if pod.get("id") != expected_id:
        raise ValueError("RunPod response ID differs from configured Pod ID")
    desired = str(pod.get("desiredStatus", "")).upper()
    direct_status = str(pod.get("status", "")).upper()
    runtime = pod.get("runtime")
    runtime_status = str((runtime or {}).get("status", "")).upper() if isinstance(runtime, dict) else ""
    if desired in {"EXITED", "STOPPED"} or direct_status in {"EXITED", "STOPPED"} \
            or runtime_status in {"EXITED", "STOPPED"}:
        return "STOPPED"
    # Current RunPod v2 `/pods/{id}` uses top-level status (RUNNING/EXITED);
    # desiredStatus is absent and runtime.status can be null on that response.
    if direct_status == "RUNNING":
        return "RUNNING"
    if direct_status in {"CREATED", "STARTING", "RESTARTING", "RESUMING"}:
        return "STARTING"
    if desired in {"RUNNING", "RESUMED"} and isinstance(runtime, dict):
        if runtime_status and runtime_status not in {"RUNNING", "READY"}:
            return "STARTING"
        return "RUNNING"
    if desired in {"RUNNING", "RESUMED", "CREATED"}:
        return "STARTING"
    raise ValueError(f"Unrecognized Pod status: desired={desired!r}, runtime={runtime_status!r}")


def get_pod(config: dict, token: str, pod: dict) -> tuple[dict, str]:
    payload = _unwrap_pod(request_api(config, token, f"/pods/{pod['id']}"))
    return payload, pod_status(payload, pod["id"])


def get_ssh(pod_config: dict, pod_data: dict) -> dict | None:
    # Current v2 responses may expose direct SSH as `ssh.direct`.
    ssh = pod_data.get("ssh")
    if isinstance(ssh, dict):
        direct = ssh.get("direct")
        if isinstance(direct, dict):
            host = direct.get("host") or direct.get("ip")
            port = direct.get("port") or direct.get("publicPort")
            user = direct.get("username") or direct.get("user") or "root"
            if host and port:
                return {"host": str(host), "port": int(port), "user": str(user)}
        elif isinstance(direct, str):
            match = re.fullmatch(r"(?:ssh\s+)?(?:-p\s+(\d+)\s+)?([^@\s]+)@([^:\s]+)(?::(\d+))?", direct.strip())
            if match:
                port = match.group(1) or match.group(4) or "22"
                return {"host": match.group(3), "port": int(port), "user": match.group(2)}
    runtime = pod_data.get("runtime")
    ports = runtime.get("ports") if isinstance(runtime, dict) else None
    if isinstance(ports, list):
        for item in ports:
            private = item.get("privatePort", item.get("private", 0)) if isinstance(item, dict) else 0
            public = item.get("publicPort", item.get("public")) if isinstance(item, dict) else None
            if isinstance(item, dict) and int(private or 0) == 22:
                if item.get("ip") and public:
                    return {"host": str(item["ip"]), "port": int(public), "user": "root"}
    configured = pod_config.get("ssh")
    if configured and configured.get("host") and configured.get("port"):
        return {"host": configured["host"], "port": int(configured["port"]),
                "user": configured.get("user", "root")}
    return None


REMOTE_SNAPSHOT = r'''set -eu
python3 - "$DURABLE_ROOT" "$STUDY_PREFIXES_JSON" "$ROLE" "$FAILED_ATTEMPT_CAP" <<'PY'
import hashlib, json, os, pathlib, re, subprocess, sys
root=pathlib.Path(sys.argv[1]); prefixes=json.loads(sys.argv[2]); role=sys.argv[3]; cap=int(sys.argv[4])
errors=[]; pending=[]; completed=[]; capped=[]; active=[]; studies=[]; dev=0; final=0
def safe(rel):
    p=pathlib.PurePosixPath(rel)
    if p.is_absolute() or not p.parts or any(x in ('', '.', '..') for x in p.parts): raise ValueError('unsafe relative path')
    out=(root/pathlib.Path(*p.parts)).resolve()
    if not out.is_relative_to(root.resolve()): raise ValueError('path escaped durable root')
    return out
def matches_result(task, result):
    fields=('task_id','dataset','experiment','backbone','training_seed','protocol_id','run_signature','optimizer_step','checkpoint_sha256')
    expected={'task_id':task.get('task_id'),'dataset':task.get('dataset'),'experiment':task.get('experiment'),
      'backbone':task.get('backbone'),'training_seed':task.get('training_seed'),'protocol_id':task.get('protocol_id'),
      'run_signature':task.get('run_signature'),'optimizer_step':task.get('optimizer_step'),
      'checkpoint_sha256':(task.get('checkpoint') or {}).get('sha256')}
    if any(result.get(k)!=expected[k] for k in fields): return False
    ev=task.get('evaluation') or {}
    def fingerprints(rows):
      return all(isinstance(r.get('initial_state_kind'),str) and r.get('initial_state_kind')
        and isinstance(r.get('initial_state_sha256'),str) and re.fullmatch(r'[0-9a-f]{64}',r['initial_state_sha256']) for r in rows)
    if task.get('dataset')=='aloha':
      rows=result.get('episodes')
      return isinstance(rows,list) and len(rows)==ev.get('episodes') and fingerprints(rows) and [r.get('initial_state_id') for r in rows]==list(range(ev.get('seed_offset',0),ev.get('seed_offset',0)+ev.get('episodes',0)))
    rows=result.get('results'); ids=ev.get('initial_state_ids')
    if ids is None and isinstance(ev.get('initial_state_offset'),int) and isinstance(ev.get('episodes_per_task'),int):
      ids=list(range(ev['initial_state_offset'],ev['initial_state_offset']+ev['episodes_per_task']))
    if not isinstance(rows,list) or not isinstance(ids,list): return False
    for tid in ev.get('tasks',[]):
      task_rows=[r for r in rows if r.get('task')==tid and r.get('condition')=='canonical']
      got=sorted(r.get('initial_state_id') for r in task_rows)
      if not fingerprints(task_rows): return False
      if got!=sorted(ids): return False
    return len(rows)==len(ev.get('tasks',[]))*len(ev.get('conditions',[]))*ev.get('episodes_per_task',0)
for path in sorted((root/'evaluation_tasks').rglob('*.task.json')) if (root/'evaluation_tasks').exists() else []:
    try:
      t=json.loads(path.read_text()); sid=t.get('study_id','')
      if not any(sid.startswith(x) for x in prefixes): continue
      if t.get('schema')!=1 or t.get('status')!='ready' or not t.get('task_id'): raise ValueError('invalid task schema/status/identity')
      result_path=safe(t.get('result_path','')); th=hashlib.sha256(t['task_id'].encode()).hexdigest()
      status_path=root/'evaluation_status'/(th+'.json'); status=None
      if status_path.exists():
        status=json.loads(status_path.read_text())
        if status.get('task_id')!=t['task_id'] or status.get('status') not in {'validating','evaluating','complete','failed'}: raise ValueError('malformed evaluation status')
        if status.get('status') in {'validating','evaluating'}: active.append(t['task_id'])
        if status.get('status')=='failed':
          attempt=status.get('attempt')
          if type(attempt) is not int or attempt<1: raise ValueError('failed status has malformed attempt')
          if attempt>=cap: capped.append({'task_id':t['task_id'],'attempt':attempt})
      if result_path.exists():
        result=json.loads(result_path.read_text())
        if matches_result(t,result): completed.append(t['task_id'])
        else: pending.append(t['task_id'])
      else: pending.append(t['task_id'])
      if t.get('purpose')=='development_selection': dev += int(t['task_id'] in pending)
      if t.get('purpose')=='final_test': final += int(t['task_id'] in pending)
    except Exception as exc: errors.append({'path':str(path),'error':type(exc).__name__+': '+str(exc)})
capped=[x for x in capped if x['task_id'] in pending]
base=root/'success_studies'
if base.exists():
  for p in sorted(base.glob('*/study_status.json')):
    try:
      s=json.loads(p.read_text())
      if not any(str(s.get('study_id','')).startswith(x) for x in prefixes): continue
      if s.get('study_id')!=p.parent.name or s.get('dataset_role') not in {'aloha','libero'}: raise ValueError('study status identity/role malformed')
      if s.get('status') not in {'running','final_test_enqueued','training_complete_evaluation_pending','failed'}: raise ValueError('study status value malformed')
      if not re.fullmatch(r'[0-9a-f]{64}',str(s.get('plan_id',''))): raise ValueError('study plan ID malformed')
      if type(s.get('total_policies')) is not int or s.get('total_policies')!=6 or not isinstance(s.get('completed_policies'),list) or not isinstance(s.get('pending_final_tests'),list): raise ValueError('study status policy/task fields malformed')
      if s.get('status')=='training_complete_evaluation_pending':
        lock=json.loads((p.parent/'locked_success_study.json').read_text())
        if lock.get('plan_id')!=s.get('plan_id') or lock.get('dataset_role')!=s.get('dataset_role'): raise ValueError('locked cloud run snapshot mismatch')
        done=s.get('completed_policies'); tasks=s.get('pending_final_tests')
        if not isinstance(done,list) or len(done)!=6 or not all(isinstance(x,dict) for x in done) or not isinstance(tasks,list) or len(tasks)!=6 or not all(isinstance(x,str) for x in tasks) or len(set(tasks))!=6: raise ValueError('completion status lacks six unique policy/task records')
        verified=False
        if os.environ.get('VERIFY_TRAIN_SNAPSHOT')=='1' and role.startswith('train_') and role.removeprefix('train_')==s.get('dataset_role'):
          for tid in tasks:
            found=False
            for tp in (root/'evaluation_tasks').rglob('*.task.json'):
              t=json.loads(tp.read_text())
              if t.get('task_id')==tid and t.get('study_id')==s['study_id'] and t.get('purpose')=='final_test' and t.get('dataset')==s.get('dataset_role'):
                cp=safe((t.get('checkpoint') or {}).get('path','')); meta=t['checkpoint']
                if not cp.is_file() or cp.stat().st_size!=meta.get('size_bytes'): raise ValueError('final checkpoint missing/size mismatch')
                h=hashlib.sha256()
                with cp.open('rb') as f:
                  for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
                if h.hexdigest()!=meta.get('sha256'): raise ValueError('final checkpoint SHA mismatch')
                found=True; break
            if not found: raise ValueError('published final task/checkpoint missing')
          verified=True
      studies.append({'study_id':s.get('study_id'),'dataset_role':s.get('dataset_role'),'status':s.get('status'),
        'completed_policies':len(s.get('completed_policies',[])) if isinstance(s.get('completed_policies'),list) else None,
        'pending_final_tests':len(s.get('pending_final_tests',[])) if isinstance(s.get('pending_final_tests'),list) else None,
        'cloud_snapshot_verified':verified})
    except Exception as exc: errors.append({'path':str(p),'error':type(exc).__name__+': '+str(exc)})
try:
  out=subprocess.check_output(['ps','-eo','pid=,ppid=,args='],text=True,timeout=10)
  procs=[]
  for line in out.splitlines():
    m=re.match(r'\s*(\d+)\s+(\d+)\s+(.*)',line)
    if not m: continue
    pid,ppid,cmd=int(m.group(1)),int(m.group(2)),m.group(3)
    if any(x in cmd for x in ('scripts/runpod_success_study.py','scripts/runpod_update_matched_study.py','scripts/train.py','scripts/run_study.py')): procs.append({'pid':pid,'ppid':ppid,'cmd':cmd[:240]})
  train_procs=[p for p in procs if 'train.py' in p['cmd'] or 'run_study.py' in p['cmd']]
  role_procs=[p for p in procs if 'runpod_success_study.py' in p['cmd'] or 'runpod_update_matched_study.py' in p['cmd']]
  sim_children=[]; eval_worker=[]
  for line in out.splitlines():
    m=re.match(r'\s*(\d+)\s+(\d+)\s+(.*)',line)
    if not m: continue
    p={'pid':int(m.group(1)),'ppid':int(m.group(2)),'cmd':m.group(3)[:240]}
    if any(x in p['cmd'] for x in ('scripts/evaluate.py','scripts/eval_libero.py')): sim_children.append(p)
    if 'scripts/runpod_eval_worker.py' in p['cmd']: eval_worker.append(p)
except Exception as exc:
  errors.append({'path':'ps','error':type(exc).__name__+': '+str(exc)}); train_procs=[{'error':'unknown'}]; role_procs=[{'error':'unknown'}]; sim_children=[{'error':'unknown'}]; eval_worker=[]
print(json.dumps({'schema':1,'role':role,'pending_task_ids':pending,'completed_task_count':len(completed),
 'pending_count':len(pending),'development_pending':dev,'final_pending':final,'active_task_ids':active,
 'capped_failed_tasks':capped,'study_statuses':studies,'errors':errors,
 'owned_training_processes':train_procs+role_procs,'simulation_children':sim_children,
 'evaluator_workers':eval_worker},separators=(',',':')))
PY
'''


def ssh_snapshot(config: dict, pod_cfg: dict, pod_data: dict, role: str,
                 verify_training: bool = False) -> tuple[dict | None, str | None]:
    access = get_ssh(pod_cfg, pod_data)
    if not access:
        return None, "SSH endpoint not present in current Pod API response"
    ssh_cfg = config["ssh"]
    key = Path(ssh_cfg["private_key"])
    if not key.is_file():
        return None, "SSH private key path does not exist"
    remote_vars = {
        "DURABLE_ROOT": config["durable_root"],
        "STUDY_PREFIXES_JSON": json.dumps(config["study_id_prefixes"], separators=(",", ":")),
        "ROLE": role,
        "FAILED_ATTEMPT_CAP": str(config["failed_attempt_cap"]),
        "VERIFY_TRAIN_SNAPSHOT": "1" if verify_training else "0",
    }
    remote_command = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in remote_vars.items()) + " bash -s"
    cmd = ["ssh", "-i", str(key), "-p", str(access["port"])]
    cmd += [f"-o{option}" for option in ssh_cfg["options"]]
    cmd += [f"{access['user']}@{access['host']}", remote_command]
    try:
        proc = subprocess.run(cmd, input=REMOTE_SNAPSHOT, text=True, capture_output=True,
                              timeout=55, check=False)
    except subprocess.TimeoutExpired:
        return None, "SSH snapshot exceeded 55 seconds"
    if proc.returncode:
        return None, f"SSH snapshot exited {proc.returncode}: {proc.stderr[-300:].strip()}"
    try:
        rows = [line for line in proc.stdout.splitlines() if line.strip()]
        snapshot = json.loads(rows[-1])
        if snapshot.get("schema") != 1 or snapshot.get("role") != role:
            raise ValueError("snapshot schema/role mismatch")
        if not isinstance(snapshot.get("errors"), list) or not isinstance(snapshot.get("pending_task_ids"), list):
            raise ValueError("snapshot arrays missing")
        return snapshot, None
    except (IndexError, ValueError, json.JSONDecodeError) as exc:
        return None, f"SSH snapshot malformed: {type(exc).__name__}: {exc}"


def remote_exec(config: dict, pod_cfg: dict, pod_data: dict, script: str,
                timeout: int = 55) -> tuple[dict | None, str | None]:
    access = get_ssh(pod_cfg, pod_data)
    if not access:
        return None, "SSH endpoint not present in current Pod API response"
    key = Path(config["ssh"]["private_key"])
    cmd = ["ssh", "-i", str(key), "-p", str(access["port"])]
    cmd += [f"-o{o}" for o in config["ssh"]["options"]]
    cmd += [f"{access['user']}@{access['host']}", "bash -s"]
    try:
        proc = subprocess.run(cmd, input=script, text=True, capture_output=True,
                              timeout=min(timeout, CAP), check=False)
    except subprocess.TimeoutExpired:
        return None, f"SSH command exceeded {min(timeout, CAP)} seconds"
    if proc.returncode:
        return None, f"SSH command exited {proc.returncode}: {proc.stderr[-300:].strip()}"
    try:
        rows = [line for line in proc.stdout.splitlines() if line.strip()]
        return json.loads(rows[-1]), None
    except (IndexError, json.JSONDecodeError) as exc:
        return None, f"SSH command returned malformed JSON: {type(exc).__name__}"


def _profile(config: dict, name: str, archive_override: str | None,
             sha_override: str | None) -> tuple[str, str]:
    profile = config["source_profiles"].get(name)
    archive = archive_override or (profile.get("archive") if isinstance(profile, dict) else None)
    sha = sha_override or (profile.get("sha256") if isinstance(profile, dict) else None)
    if not archive or not sha or not re.fullmatch(r"[0-9a-f]{64}", str(sha)):
        raise ValueError(f"Source profile {name!r} needs an archive path and lowercase SHA-256")
    if not str(archive).startswith("/") or any(c in str(archive) for c in "\n\r;`$"):
        raise ValueError("Source archive must be a safe absolute cloud-volume path")
    return str(archive), str(sha)


def bootstrap_script(config: dict, archive: str, sha: str) -> str:
    root = shlex.quote(config["repo_root"])
    archive_q = shlex.quote(archive)
    return f'''set -eu
ROOT={root}
ARCHIVE={archive_q}
EXPECTED={sha}
mkdir -p "$ROOT" /workspace-global/action-omni/workers/evaluation
if pgrep -af '[s]cripts/runpod_eval_worker.py' >/dev/null; then
  if [ "$(cat "$ROOT/.runpod_eval_source_sha" 2>/dev/null || true)" != "$EXPECTED" ]; then
    echo '{{"setup":"source_mismatch_worker_active"}}'; exit 0
  fi
  echo '{{"setup":"ready","worker":"already_running"}}'; exit 0
fi
if pgrep -af '[s]etup_runpod_eval.sh' >/dev/null; then
  echo '{{"setup":"running"}}'; exit 0
fi
if [ "$(cat "$ROOT/.runpod_eval_source_sha" 2>/dev/null || true)" != "$EXPECTED" ]; then
python3 - "$ARCHIVE" "$EXPECTED" "$ROOT" <<'PY'
import hashlib, pathlib, shutil, sys, tarfile, tempfile
archive=pathlib.Path(sys.argv[1]); expected=sys.argv[2]; root=pathlib.Path(sys.argv[3])
if not archive.is_file(): raise SystemExit('source archive not found on shared volume')
h=hashlib.sha256()
with archive.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
if h.hexdigest()!=expected: raise SystemExit('source archive SHA-256 mismatch')
tmp=pathlib.Path(tempfile.mkdtemp(prefix='action-omni-src-'))
try:
  with tarfile.open(archive,'r:gz') as tf:
    members=tf.getmembers()
    for m in members:
      p=pathlib.PurePosixPath(m.name)
      if p.is_absolute() or any(x in ('','..') for x in p.parts) or not (m.isdir() or m.isfile()):
        raise SystemExit('source archive contains unsafe member')
    tf.extractall(tmp, members=members)
  source=tmp
  if (tmp/'action-omni').is_dir(): source=tmp/'action-omni'
  elif len(list(tmp.iterdir()))==1 and next(tmp.iterdir()).is_dir(): source=next(tmp.iterdir())
  if not (source/'scripts'/'setup_runpod_eval.sh').is_file() or not (source/'scripts'/'runpod_eval_worker.py').is_file():
    raise SystemExit('source archive lacks evaluator bootstrap files')
  shutil.copytree(source,root,dirs_exist_ok=True)
finally: shutil.rmtree(tmp,ignore_errors=True)
PY
  printf '%s\\n' "$EXPECTED" > "$ROOT/.runpod_eval_source_sha"
  rm -f "$ROOT/.runpod_eval_setup_failed"
fi
SETUP_SHA="$(sha256sum "$ROOT/scripts/setup_runpod_eval.sh" | cut -d' ' -f1)"
if [ -f "$ROOT/.runpod_eval_setup_failed" ] && [ "$(cat "$ROOT/.runpod_eval_setup_sha" 2>/dev/null || true)" != "$SETUP_SHA" ]; then
  rm -f "$ROOT/.runpod_eval_setup_failed"
fi
if [ -f "$ROOT/.runpod_eval_setup_failed" ]; then
  echo '{{"setup":"failed","reason":"inspect shared evaluation setup.log; clear failure marker only after remediation"}}'; exit 0
fi
if [ ! -x "$ROOT/.venv-eval-aloha/bin/python" ] || [ ! -x "$ROOT/.venv-eval-libero/bin/python" ] || [ "$(cat "$ROOT/.runpod_eval_setup_sha" 2>/dev/null || true)" != "$SETUP_SHA" ]; then
  if pgrep -af '[s]etup_runpod_eval.sh' >/dev/null; then
    echo '{{"setup":"running"}}'
    exit 0
  fi
  LOG=/workspace-global/action-omni/workers/evaluation/setup.log
  nohup flock -n "$ROOT/.eval_setup.lock" bash -lc 'cd "$1"; if bash scripts/setup_runpod_eval.sh; then printf "%s\\n" "$2" > .runpod_eval_setup_sha; rm -f .runpod_eval_setup_failed; else rc=$?; printf "%s\\n" "$rc" > .runpod_eval_setup_failed; exit "$rc"; fi' _ "$ROOT" "$SETUP_SHA" >>"$LOG" 2>&1 </dev/null &
  echo '{{"setup":"started"}}'
else
  echo '{{"setup":"ready"}}'
fi
'''


def launch_worker_script(config: dict, poll_seconds: int) -> str:
    root = shlex.quote(config["repo_root"])
    durable = shlex.quote(config["durable_root"])
    log = "/workspace-global/action-omni/workers/evaluation/worker.log"
    return f'''set -eu
ROOT={root}; DURABLE={durable}; LOG={shlex.quote(log)}
mkdir -p "$(dirname "$LOG")"
if pgrep -af '[s]cripts/runpod_eval_worker.py' >/dev/null; then
  echo '{{"worker":"already_running"}}'; exit 0
fi
nohup flock -n "$ROOT/.eval_worker.lock" env MUJOCO_GL=egl PYTHONUNBUFFERED=1 python3 "$ROOT/scripts/runpod_eval_worker.py" \\
  --durable-root "$DURABLE" --work-root "$ROOT/asset/eval_worker" \\
  --aloha-python "$ROOT/.venv-eval-aloha/bin/python" \\
  --libero-python "$ROOT/.venv-eval-libero/bin/python" \\
  --poll-seconds {int(poll_seconds)} >>"$LOG" 2>&1 </dev/null &
echo '{{"worker":"started","pid":'"$!"'}}'
'''


def stop_sim_pod(config: dict, token: str, pod_cfg: dict, pod_data: dict,
                 dry_run: bool) -> tuple[str, dict | None]:
    before, status = get_pod(config, token, pod_cfg)
    if status != "RUNNING":
        return status, before
    # State is queried immediately before mutation. The remote gate is rechecked by caller.
    if dry_run:
        return "WOULD_STOP", before
    request_api(config, token, f"/pods/{pod_cfg['id']}/stop", "POST")
    after, after_status = get_pod(config, token, pod_cfg)
    if after_status not in {"STOPPED", "STARTING"}:
        raise RuntimeError(f"Pod stop did not produce a stopped/transition state ({after_status})")
    return after_status, after


def start_sim_pod(config: dict, token: str, pod_cfg: dict,
                  dry_run: bool) -> tuple[str, dict]:
    before, status = get_pod(config, token, pod_cfg)
    if status == "RUNNING":
        return status, before
    if status == "STARTING":
        return status, before
    if status != "STOPPED":
        raise RuntimeError(f"Refusing to start Pod in unrecognized state {status}")
    if dry_run:
        return "WOULD_START", before
    request_api(config, token, f"/pods/{pod_cfg['id']}/start", "POST")
    after, after_status = get_pod(config, token, pod_cfg)
    if after_status not in {"RUNNING", "STARTING"}:
        raise RuntimeError(f"Pod start did not produce running/transition state ({after_status})")
    return after_status, after


def trainer_stop_ready(pod: dict, snapshot: dict) -> tuple[bool, str]:
    if snapshot.get("errors"):
        return False, "cloud snapshot has errors"
    if snapshot.get("owned_training_processes"):
        return False, "owned training processes remain or process state is unknown"
    matching = [s for s in snapshot.get("study_statuses", [])
                if s.get("dataset_role") == pod["role"].removeprefix("train_")
                and s.get("status") == "training_complete_evaluation_pending"]
    if len(matching) != 1:
        return False, "exactly one matching training_complete_evaluation_pending status is required"
    status = matching[0]
    if (status.get("completed_policies") != 6 or status.get("pending_final_tests") != 6
            or status.get("cloud_snapshot_verified") is not True):
        return False, "six policy/checkpoint/task snapshots are not verified"
    return True, "completion status, zero process count, and durable checkpoint/task snapshot verified"


def manager_tick(config: dict, token: str, args, state: dict) -> tuple[dict, dict]:
    pods = config["pods"]
    tick = {"utc": utc_now(), "snapshots": {}, "actions": [], "errors": []}
    pod_data: dict[str, dict] = {}
    pod_statuses: dict[str, str] = {}
    for name, pod in pods.items():
        try:
            data, status = get_pod(config, token, pod)
            pod_data[name], pod_statuses[name] = data, status
            role = pod["role"]
            if status in {"RUNNING", "STARTING"}:
                snap, error = ssh_snapshot(config, pod, data, role,
                                           verify_training=args.stop_complete_trainers and role.startswith("train_"))
                if error:
                    tick["errors"].append({"pod": name, "error": error})
                elif snap is not None:
                    snap["pod_status"] = status
                    tick["snapshots"][name] = snap
            else:
                tick["snapshots"][name] = {"role": role, "pod_status": status,
                                            "pending_count": None, "errors": []}
        except Exception as exc:
            tick["errors"].append({"pod": name, "error": f"{type(exc).__name__}: {exc}"})

    eval_pod = pods["evaluation"]
    eval_snap = tick["snapshots"].get("evaluation")
    queue_snapshots = [snap for snap in tick["snapshots"].values()
                       if type(snap.get("pending_count")) is int and not snap.get("errors")]
    pending_sets = {tuple(sorted(snap.get("pending_task_ids", []))) for snap in queue_snapshots}
    if eval_pod["id"] in pod_data and pod_statuses.get("evaluation") == "RUNNING" and eval_snap is None:
        # A failed evaluator SSH snapshot is a hard hold; do not infer an empty queue.
        tick["evaluation_gate"] = "hold_snapshot_unavailable"
    elif eval_snap is not None and eval_snap.get("errors"):
        tick["evaluation_gate"] = "hold_malformed_or_unverified_cloud_state"
    elif len(pending_sets) > 1:
        tick["evaluation_gate"] = "hold_shared_queue_snapshots_disagree"
        tick["errors"].append({"pod": "shared_volume", "error": "reachable Pod snapshots disagree on pending task IDs"})
    elif not queue_snapshots:
        tick["evaluation_gate"] = "hold_no_verified_queue_snapshot"
    else:
        # Any running role can read the shared queue, including while the evaluator Pod is stopped.
        queue_snap = eval_snap if eval_snap and type(eval_snap.get("pending_count")) is int else queue_snapshots[0]
        pending = queue_snap.get("pending_count")
        if type(pending) is not int:
            tick["evaluation_gate"] = "hold_pending_count_malformed"
        elif queue_snap.get("capped_failed_tasks"):
            state["evaluation_idle_since"] = None
            tick["evaluation_gate"] = "hold_capped_failed_tasks"
            tick["errors"].append({"pod": "evaluation", "error": "failed task reached configured attempt cap",
                                    "tasks": queue_snap["capped_failed_tasks"]})
        elif pending > 0:
            state["evaluation_idle_since"] = None
            tick["evaluation_gate"] = "pending_work"
            try:
                status, data = start_sim_pod(config, token, eval_pod, args.dry_run)
                tick["actions"].append({"pod": "evaluation", "action": status.lower(),
                                        "reason": f"{pending} verified/incomplete task result(s)"})
                if args.dry_run and status == "RUNNING":
                    tick["actions"].append({"pod": "evaluation", "action": "would_ensure_bootstrap_worker"})
                elif status == "RUNNING":
                    ssh_error = None
                    if status == "RUNNING":
                        archive, sha = _profile(config, args.source_set, args.source_archive, args.source_sha256)
                        boot, ssh_error = remote_exec(config, eval_pod, data,
                                                      bootstrap_script(config, archive, sha))
                        if ssh_error:
                            tick["errors"].append({"pod": "evaluation", "error": ssh_error})
                        elif boot and boot.get("setup") == "source_mismatch_worker_active":
                            tick["errors"].append({"pod": "evaluation", "error": "worker active with a different immutable source archive; refusing overlay"})
                        elif boot and boot.get("setup") == "failed":
                            tick["errors"].append({"pod": "evaluation", "error": boot.get("reason", "simulator setup failed; inspect evaluation setup log")})
                        elif boot and boot.get("setup") == "ready":
                            launch, launch_error = remote_exec(config, eval_pod, data,
                                                               launch_worker_script(config, config["poll_seconds"]))
                            if launch_error:
                                tick["errors"].append({"pod": "evaluation", "error": launch_error})
                            else:
                                tick["actions"].append({"pod": "evaluation", "action": "worker_ensured",
                                                        "worker": launch})
                        else:
                            tick["actions"].append({"pod": "evaluation", "action": "setup_ensured",
                                                    "setup": boot})
            except Exception as exc:
                tick["errors"].append({"pod": "evaluation", "error": f"{type(exc).__name__}: {exc}"})
        else:
            tick["evaluation_gate"] = "empty_queue"
            if (pod_statuses.get("evaluation") == "RUNNING" and eval_snap is not None
                    and not eval_snap.get("simulation_children")):
                now = time.time()
                idle = state.get("evaluation_idle_since")
                if idle is None:
                    state["evaluation_idle_since"] = now
                    tick["actions"].append({"pod": "evaluation", "action": "idle_grace_started"})
                elif now - float(idle) >= config["idle_grace_seconds"]:
                    # Re-read both cloud queue and Pod state directly before a stop.
                    fresh, error = ssh_snapshot(config, eval_pod, pod_data.get("evaluation", {}), "evaluation")
                    if error or not fresh or fresh.get("errors") or fresh.get("pending_count") != 0:
                        tick["errors"].append({"pod": "evaluation", "error": error or "queue changed or failed closed on stop recheck"})
                    elif fresh.get("simulation_children"):
                        tick["actions"].append({"pod": "evaluation", "action": "stop_deferred_simulation_child_active"})
                    elif not args.dry_run:
                        latest, latest_status = get_pod(config, token, eval_pod)
                        if latest_status != "RUNNING":
                            tick["actions"].append({"pod": "evaluation", "action": "stop_skipped_status_changed",
                                                    "status": latest_status})
                        else:
                            status, _ = stop_sim_pod(config, token, eval_pod, latest, args.dry_run)
                            tick["actions"].append({"pod": "evaluation", "action": status.lower(),
                                                    "reason": "empty queue grace elapsed; no simulation child"})
                            state["evaluation_idle_since"] = None
                    else:
                        tick["actions"].append({"pod": "evaluation", "action": "would_stop",
                                                "reason": "empty queue grace elapsed; no simulation child"})
            elif eval_snap is not None and eval_snap.get("simulation_children"):
                state["evaluation_idle_since"] = None
                tick["actions"].append({"pod": "evaluation", "action": "stop_deferred_simulation_child_active"})

    if args.stop_complete_trainers:
        for name in ("trainaloha", "trainlibero"):
            pod = pods[name]
            snap = tick["snapshots"].get(name)
            if pod_statuses.get(name) != "RUNNING" or snap is None:
                continue
            ready, why = trainer_stop_ready(pod, snap)
            if ready:
                try:
                    before, status = get_pod(config, token, pod)
                    fresh, err = ssh_snapshot(config, pod, before, pod["role"], verify_training=True)
                    ready2, why2 = trainer_stop_ready(pod, fresh or {}) if not err else (False, err)
                    if ready2:
                        stop_status, _ = stop_sim_pod(config, token, pod, before, args.dry_run)
                        tick["actions"].append({"pod": name, "action": stop_status.lower(), "reason": why2})
                    else:
                        tick["actions"].append({"pod": name, "action": "stop_deferred", "reason": why2})
                except Exception as exc:
                    tick["errors"].append({"pod": name, "error": f"{type(exc).__name__}: {exc}"})
            else:
                tick["actions"].append({"pod": name, "action": "trainer_stop_not_authorized_by_state",
                                        "reason": why})
    return tick, state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--source-set", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--source-archive", help="Override cloud archive path selected by --source-set")
    parser.add_argument("--source-sha256", help="Override expected lowercase SHA-256")
    parser.add_argument("--once", action="store_true", help="One management tick, then exit")
    parser.add_argument("--dry-run", action="store_true", help="Read-only: report planned API changes")
    parser.add_argument("--stop-complete-trainers", action="store_true",
                        help="Opt in to strict status/process/snapshot-gated trainer stops")
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if config.get("schema") != 1 or set(config.get("pods", {})) != {"trainaloha", "trainlibero", "evaluation"}:
            raise ValueError("Pod config schema or exact Pod roles are invalid")
        if not args.dry_run:
            _profile(config, args.source_set, args.source_archive, args.source_sha256)
        state = read_json(args.state_file, {"schema": 1, "evaluation_idle_since": None,
                                            "api_error_count": 0, "next_api_retry_epoch": 0})
        if state.get("schema") != 1:
            raise ValueError("State file schema is invalid; refusing unsafe recovery")
        token = api_token()
    except Exception as exc:
        emit("configuration_error", error=f"{type(exc).__name__}: {exc}")
        return 2
    while True:
        now = time.time()
        if now < float(state.get("next_api_retry_epoch", 0)):
            emit("backoff", retry_in_s=round(state["next_api_retry_epoch"] - now, 1))
            if args.once:
                return 0
            time.sleep(min(config["poll_seconds"], max(1, state["next_api_retry_epoch"] - now)))
            continue
        tick = None
        try:
            tick, state = manager_tick(config, token, args, state)
            if tick.get("errors"):
                count = int(state.get("api_error_count", 0)) + 1
                base = int(config["api_backoff_seconds"]["base"])
                maximum = int(config["api_backoff_seconds"]["max"])
                delay = min(maximum, base * (2 ** min(count - 1, 20)))
                state["api_error_count"] = count
                state["next_api_retry_epoch"] = time.time() + delay
                tick["backoff_seconds"] = delay
            else:
                state["api_error_count"] = 0
                state["next_api_retry_epoch"] = 0
            append_log(args.log_file, tick)
            write_json_atomic(args.state_file, state)
            # Keep stdout compact and never emit Pod payloads or credentials.
            emit("tick", evaluation_gate=tick.get("evaluation_gate"),
                 pod_statuses={name: snap.get("pod_status", "reachable")
                               for name, snap in tick.get("snapshots", {}).items()},
                 queue={name: {"pending": snap.get("pending_count"),
                               "completed": snap.get("completed_task_count"),
                               "development_pending": snap.get("development_pending"),
                               "final_pending": snap.get("final_pending")}
                        for name, snap in tick.get("snapshots", {}).items()},
                 actions=tick.get("actions", []), errors=tick.get("errors", []))
        except Exception as exc:
            count = int(state.get("api_error_count", 0)) + 1
            base = int(config["api_backoff_seconds"]["base"])
            maximum = int(config["api_backoff_seconds"]["max"])
            delay = min(maximum, base * (2 ** min(count - 1, 20)))
            state["api_error_count"] = count
            state["next_api_retry_epoch"] = time.time() + delay
            write_json_atomic(args.state_file, state)
            row = emit("manager_error_backoff", count=count, delay_s=delay,
                       error=f"{type(exc).__name__}: {exc}")
            append_log(args.log_file, row)
        if args.once:
            return 1 if tick is None or tick.get("errors") else 0
        time.sleep(max(1, int(config["poll_seconds"])))


if __name__ == "__main__":
    raise SystemExit(main())
