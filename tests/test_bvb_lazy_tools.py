"""Focused tests for the BVB lazy archive backend and MCP compatibility path."""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
import sys
import tarfile
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.bvb.config import DEFAULT_RELEASE_DIR, MCP_PROFILE_PATH

MCP_ROOT = Path(os.environ.get("BVB_TEST_RELEASE", DEFAULT_RELEASE_DIR)) / MCP_PROFILE_PATH
if not MCP_ROOT.is_dir():
    raise unittest.SkipTest(f"Fetch the release first; MCP profile not found at {MCP_ROOT}")
sys.dont_write_bytecode = True
sys.path.insert(0, str(MCP_ROOT))

from tools.archive_store import ArchiveStore, ArchiveStoreError, get_archive_store
from adapters.bvb.tool_archive_index import (
    ToolArchiveIndexError,
    validate_tool_archive_index,
)


def _stable_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_archive(path: Path, files: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as tar:
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            member.mtime = 0
            tar.addfile(member, io.BytesIO(data))


def _index_for(archive_root: Path, archives: dict[str, dict[str, bytes]]) -> dict:
    archive_records = []
    members = {}
    for relative, files in archives.items():
        path = archive_root / relative
        _write_archive(path, files)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        roots = {name.split("/", 1)[0] for name in files}
        logical_root = next(iter(roots)) + "/"
        archive_records.append(
            {
                "path": relative,
                "sha256": digest,
                "size_bytes": path.stat().st_size,
                "format": "tar.gz",
                "logical_root": logical_root,
                "member_count": len(files),
                "tool_member_count": len(files),
                "non_tool_member_count": 0,
            }
        )
        for name, data in files.items():
            members[name] = {
                "archive_path": relative,
                "member_path": name,
                "size_bytes": len(data),
                "sha256": _sha256(data),
            }
    archive_set = hashlib.sha256()
    for archive in sorted(archive_records, key=lambda item: item["path"]):
        archive_set.update(archive["path"].encode())
        archive_set.update(b"\0")
        archive_set.update(archive["sha256"].encode())
        archive_set.update(b"\n")
    index = {
        "schema_version": "bvb-tool-archive-index/v1",
        "tool_profile_source_tree_sha256": "0" * 64,
        "archive_set_sha256": archive_set.hexdigest(),
        "archives": archive_records,
        "members": members,
        "non_tool_members": {},
        "api_indexes": {
            "market_data_platform": {
                "tickers": ["AAPL-US"],
                "files_by_ticker": {
                    "AAPL-US": ["price_history.xlsx"],
                    "_wacc": ["betas_for_wacc.xlsx"],
                },
            },
            "company_brand_assets": {
                "logos": [
                    {
                        "logical_path": "company_brand_assets/AAPL.png",
                        "filename": "AAPL.png",
                        "stem": "AAPL",
                        "ticker": "AAPL",
                        "company": "Apple Inc.",
                    }
                ]
            },
            "sec_edgar": {
                "cik_lookup": "sec_edgar/data/cik-lookup.json",
                "submissions": {
                    "0000320193": "sec_edgar/data/submissions/0000320193.json"
                },
                "company_facts": {},
                "frames": {},
                "filings": {
                    "0000320193": {
                        "0000320193-25-000079": (
                            "sec_edgar/data/filings/0000320193/"
                            "0000320193-25-000079.md"
                        )
                    }
                },
            },
        },
    }
    payload = dict(index)
    index["index_sha256"] = hashlib.sha256(_stable_json_bytes(payload)).hexdigest()
    return index


def _write_index(path: Path, index: dict) -> None:
    path.write_bytes(_stable_json_bytes(index))


class FakeMCP:
    def __init__(self) -> None:
        self.tools = {}

    def tool(self):
        def register(function):
            self.tools[function.__name__] = function
            return function

        return register


class ArchiveStoreTests(unittest.TestCase):
    def test_index_validation_rejects_traversal_and_digest_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_root = root / "archives"
            index = _index_for(
                archive_root,
                {
                    "shared-tools/market.tar.gz": {
                        "market_data_platform/A-US/one.bin": b"one"
                    }
                },
            )
            validate_tool_archive_index(index)

            traversal = json.loads(json.dumps(index))
            member = traversal["members"].pop(
                "market_data_platform/A-US/one.bin"
            )
            traversal["members"]["../escape"] = member
            payload = dict(traversal)
            payload.pop("index_sha256")
            traversal["index_sha256"] = hashlib.sha256(
                _stable_json_bytes(payload)
            ).hexdigest()
            with self.assertRaisesRegex(ToolArchiveIndexError, "[Uu]nsafe"):
                validate_tool_archive_index(traversal)

            corrupted = json.loads(json.dumps(index))
            corrupted["archives"][0]["sha256"] = "f" * 64
            with self.assertRaisesRegex(ToolArchiveIndexError, "self digest"):
                validate_tool_archive_index(corrupted)

    def test_cache_hit_corruption_repair_eviction_and_concurrency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_root = root / "archives"
            index = _index_for(
                archive_root,
                {
                    "shared-tools/market.tar.gz": {
                        "market_data_platform/A-US/one.bin": b"111111",
                        "market_data_platform/B-US/two.bin": b"222222",
                    }
                },
            )
            index_path = root / "index.json"
            _write_index(index_path, index)
            store = ArchiveStore(
                index_path=index_path,
                archive_root=archive_root,
                cache_root=root / "cache",
                cache_max_bytes=6,
            )

            first = store.materialize("market_data_platform/A-US/one.bin")
            self.assertEqual(first.read_bytes(), b"111111")
            first.write_bytes(b"broken")
            self.assertEqual(store.materialize("market_data_platform/A-US/one.bin").read_bytes(), b"111111")

            barrier = threading.Barrier(4)

            def materialize() -> bytes:
                barrier.wait()
                return store.read_bytes("market_data_platform/A-US/one.bin")

            with ThreadPoolExecutor(max_workers=4) as pool:
                self.assertEqual(list(pool.map(lambda _: materialize(), range(4))), [b"111111"] * 4)

            second = store.materialize("market_data_platform/B-US/two.bin")
            self.assertEqual(second.read_bytes(), b"222222")
            self.assertFalse(first.exists())
            self.assertLessEqual(
                sum(path.stat().st_size for path in (root / "cache" / "members").rglob("*") if path.is_file()),
                6,
            )

    def test_rejects_cache_under_archives_and_detects_archive_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_root = root / "archives"
            index = _index_for(
                archive_root,
                {
                    "shared-tools/market.tar.gz": {
                        "market_data_platform/A-US/one.bin": b"one"
                    }
                },
            )
            index_path = root / "index.json"
            _write_index(index_path, index)
            with self.assertRaisesRegex(ArchiveStoreError, "outside the immutable"):
                ArchiveStore(
                    index_path=index_path,
                    archive_root=archive_root,
                    cache_root=archive_root / "cache",
                    cache_max_bytes=10,
                )

            archive_path = archive_root / "shared-tools" / "market.tar.gz"
            archive_path.write_bytes(archive_path.read_bytes() + b"corrupt")
            store = ArchiveStore(
                index_path=index_path,
                archive_root=archive_root,
                cache_root=root / "cache",
                cache_max_bytes=10,
            )
            with self.assertRaisesRegex(ArchiveStoreError, "size mismatch"):
                store.materialize("market_data_platform/A-US/one.bin")

    def test_member_larger_than_budget_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_root = root / "archives"
            index = _index_for(
                archive_root,
                {
                    "shared-tools/market.tar.gz": {
                        "market_data_platform/A-US/one.bin": b"oversized"
                    }
                },
            )
            index_path = root / "index.json"
            _write_index(index_path, index)
            store = ArchiveStore(
                index_path=index_path,
                archive_root=archive_root,
                cache_root=root / "cache",
                cache_max_bytes=3,
            )
            with self.assertRaisesRegex(ArchiveStoreError, "exceeding"):
                store.materialize("market_data_platform/A-US/one.bin")


class LazyToolParityTests(unittest.TestCase):
    def setUp(self) -> None:
        get_archive_store.cache_clear()

    def tearDown(self) -> None:
        get_archive_store.cache_clear()

    def test_listing_and_output_bytes_match_extracted_tree(self) -> None:
        market_data = b"market workbook bytes"
        logo_data = b"png bytes"
        lookup_data = b'{"AAPL": 320193}'
        filing_data = b"# Apple filing"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_root = root / "archives"
            extracted = root / "extracted"
            files = {
                "market_data_platform/AAPL-US/price_history.xlsx": market_data,
                "company_brand_assets/AAPL.png": logo_data,
                "sec_edgar/data/cik-lookup.json": lookup_data,
                (
                    "sec_edgar/data/filings/0000320193/"
                    "0000320193-25-000079.md"
                ): filing_data,
            }
            for relative, data in files.items():
                path = extracted / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            index = _index_for(
                archive_root,
                {
                    "shared-tools/market.tar.gz": {
                        key: value
                        for key, value in files.items()
                        if key.startswith("market_data_platform/")
                    },
                    "shared-tools/logos.tar.gz": {
                        key: value
                        for key, value in files.items()
                        if key.startswith("company_brand_assets/")
                    },
                    "shared-tools/sec.tar.gz": {
                        key: value
                        for key, value in files.items()
                        if key.startswith("sec_edgar/")
                    },
                },
            )
            index_path = root / "index.json"
            _write_index(index_path, index)
            workspace = root / "workspace"

            legacy = self._exercise_tools(
                workspace=workspace,
                environment={
                    "MARKET_DATA_PLATFORM_DATA_PATH": str(
                        extracted / "market_data_platform"
                    ),
                    "COMPANY_BRAND_ASSETS_DATA_PATH": str(
                        extracted / "company_brand_assets"
                    ),
                    "SEC_EDGAR_DATA_PATH": str(extracted / "sec_edgar"),
                },
            )
            for child in workspace.iterdir():
                child.unlink()

            lazy = self._exercise_tools(
                workspace=workspace,
                environment={
                    "BVB_TOOL_ARCHIVE_INDEX": str(index_path),
                    "BVB_TOOL_ARCHIVE_ROOT": str(archive_root),
                    "BVB_TOOL_CACHE_ROOT": str(root / "tool-cache"),
                    "BVB_TOOL_CACHE_MAX_BYTES": "1024",
                },
            )
            self.assertEqual(lazy, legacy)

    def _exercise_tools(self, *, workspace: Path, environment: dict[str, str]) -> dict:
        workspace.mkdir(exist_ok=True)
        clear = {
            "BVB_TOOL_ARCHIVE_INDEX": None,
            "BVB_TOOL_ARCHIVE_ROOT": None,
            "BVB_TOOL_CACHE_ROOT": None,
            "BVB_TOOL_CACHE_MAX_BYTES": None,
            "MARKET_DATA_PLATFORM_DATA_PATH": None,
            "COMPANY_BRAND_ASSETS_DATA_PATH": None,
            "SEC_EDGAR_DATA_PATH": None,
        }
        values = {**clear, **environment}
        configured = {key: value for key, value in values.items() if value is not None}
        with patch.dict(os.environ, configured, clear=False):
            for key, value in values.items():
                if value is None:
                    os.environ.pop(key, None)
            get_archive_store.cache_clear()
            market = importlib.import_module("tools.market_data_platform")
            logos = importlib.import_module("tools.company_brand_assets")
            sec = importlib.import_module("tools.sec_edgar")
            mcp = FakeMCP()
            with (
                patch.object(market, "validate_workspace_path", return_value=str(workspace)),
                patch.object(logos, "validate_workspace_path", return_value=str(workspace)),
                patch.object(sec, "validate_workspace_path", return_value=str(workspace)),
            ):
                market.register(mcp, 10001)
                logos.register(mcp, 10001)
                sec.register(mcp, 10001)
                tickers = mcp.tools["list_tickers"]()
                available = mcp.tools["list_available_data"]("AAPL")
                market_result = mcp.tools["get_company_data"](
                    "AAPL", str(workspace), "price_history"
                )
                logo_listing = mcp.tools["list_available_logos"]()
                logo_result = mcp.tools["get_logo_file"]("AAPL", str(workspace))
                filings = mcp.tools["list_available_filings"]("320193")
                filing_result = mcp.tools["get_filing"](
                    "320193", "0000320193-25-000079", str(workspace)
                )
            return {
                "tickers": tickers,
                "available": available,
                "market_success": market_result["success"],
                "market_bytes": (workspace / "AAPL-US Price History (Daily).xlsx").read_bytes(),
                "logo_listing": logo_listing,
                "logo_success": logo_result["success"],
                "logo_bytes": (workspace / "AAPL.png").read_bytes(),
                "filings": filings,
                "filing_success": filing_result["success"],
                "filing_bytes": (
                    workspace
                    / "filing_0000320193_0000320193-25-000079.md"
                ).read_bytes(),
            }


if __name__ == "__main__":
    unittest.main()
