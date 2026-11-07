"""Static checks on the local and Daytona runtime images."""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adapters.bvb.adapter import TEMPLATE_DIR
from adapters.bvb.daytona import _context_dockerfile

LOCAL_DOCKERFILES = (
    TEMPLATE_DIR / "environment" / "Dockerfile",
    TEMPLATE_DIR / "_environment-rewardkit" / "Dockerfile",
)


def _apt_packages(dockerfile: str) -> list[list[str]]:
    return [match.split() for match in re.findall(r"apt-get install -y ([^&\n]+)", dockerfile)]


class RuntimeImageTests(unittest.TestCase):
    def test_local_images_can_read_zstd_archives(self) -> None:
        # The MCP server shells out to zstd for the SEC .tar.zst archives.
        for path in LOCAL_DOCKERFILES:
            with self.subTest(dockerfile=path.relative_to(TEMPLATE_DIR).as_posix()):
                final_stage = _apt_packages(path.read_text(encoding="utf-8"))[-1]
                self.assertIn("zstd", final_stage)

    def test_daytona_image_installs_the_local_package_set(self) -> None:
        local = _apt_packages((TEMPLATE_DIR / "environment" / "Dockerfile").read_text(encoding="utf-8"))
        daytona = _apt_packages(
            _context_dockerfile(identity={"cache_max_bytes": 1}, include_archives=True, include_index=True)
        )
        self.assertEqual(daytona, local)


if __name__ == "__main__":
    unittest.main()
