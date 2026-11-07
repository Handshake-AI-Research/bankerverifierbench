"""Select an isolated LiteLLM overlay without altering baseline environments."""
import re
import subprocess


def pin_dependency_cache(env, version):
    """Keep package caches independent of each experiment's artifact cache."""
    if version and not env.get('UV_CACHE_DIR'):
        env['UV_CACHE_DIR'] = subprocess.check_output(
            ['uv', 'cache', 'dir'], env=env, text=True, timeout=30,
        ).strip()


def configure_worker(cmd, env, version):
    if not version: return
    if not re.fullmatch(r'\d+\.\d+\.\d+', version): raise ValueError('Invalid LiteLLM version')
    cmd[4:4]=['--no-sync','--with',f'litellm=={version}']
    env['BVB_LITELLM_VERSION']=version
