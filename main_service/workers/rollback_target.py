"""Helpers to identify unreliable clients and find rollback target from provenance."""

import json
from typing import Any, Dict, List, Optional, Tuple

import torch
from main_service.blockchain.fabric_client import FabricClient
from shared.logger import setup_logger
from shared.storage.ipfs_client import IPFSClient
from shared.storage.encryption import EncryptionService
from shared.datasets import get_dataset
from torch.utils.data import DataLoader
from main_service.workers.regression_diagnosis import RegressionDiagnosis

logger = setup_logger(__name__)


def _ipfs_cid_from_provenance(provenance: Dict[str, Any]) -> Optional[str]:
    """Extract IPFS CID from provenance (metadata or validation_history)."""
    metadata = provenance.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    cid = (
        metadata.get("ipfs_cid")
        or provenance.get("ipfs_cid")
        or provenance.get("ipfsCID")
    )
    if cid:
        return cid
    history = metadata.get("validation_history") or []
    if isinstance(history, list):
        for record in reversed(history):
            if isinstance(record, dict) and record.get("ipfs_cid"):
                return record["ipfs_cid"]
    return None


def _client_ids_from_provenance(provenance: Dict[str, Any]) -> List[str]:
    """Extract contributing client IDs from provenance metadata."""
    metadata = provenance.get("metadata") or {}
    if not isinstance(metadata, dict):
        return []
    ids = metadata.get("client_ids")
    if isinstance(ids, list):
        return [str(x) for x in ids]
    return []


async def _diagnose_by_individual_diffs(
    bad_provenance: Dict[str, Any],
    good_provenance: Dict[str, Any],
    baseline_accuracy: float,
    accuracy_threshold: float = 0.5,
) -> List[str]:
    """
    Identify problematic clients by testing each client's diff individually (leave-one-out):
    fetch parent (good) weights and each client's diff from IPFS, apply each diff alone,
    validate; flag clients that cause a significant accuracy drop.
    """
    metadata = bad_provenance.get("metadata") or {}
    if not isinstance(metadata, dict):
        return []
    client_ids = _client_ids_from_provenance(bad_provenance)
    cid_map = metadata.get("client_weight_diff_cids")
    if not isinstance(cid_map, dict) or not client_ids:
        logger.info(
            "Individual-diff diagnosis skipped: no client_weight_diff_cids or client_ids in bad version."
        )
        return []

    good_cid = _ipfs_cid_from_provenance(good_provenance)
    if not good_cid:
        logger.warning(
            "Individual-diff diagnosis skipped: no IPFS CID for good version."
        )
        return []

    try:
        encryption = EncryptionService()
    except Exception as e:
        logger.warning(
            f"Individual-diff diagnosis skipped: encryption not available: {e}"
        )
        return []

    async with IPFSClient() as ipfs_client:
        # Fetch good (parent) weights: encrypted in IPFS, decrypt and parse
        try:
            enc_bytes = await ipfs_client.get_bytes(good_cid)
            dec_bytes = encryption.decrypt_diff(enc_bytes)
            previous_weights = {
                k: torch.tensor(v)
                for k, v in json.loads(dec_bytes.decode("utf-8")).items()
            }
        except Exception as e:
            logger.warning(f"Diagnosis: could not fetch/decrypt good weights: {e}")
            return []

        # Build client_updates with weight_diff string for each client
        client_updates: List[Dict[str, Any]] = []
        for client_id in client_ids:
            diff_cid = cid_map.get(client_id) if isinstance(cid_map, dict) else None
            if not diff_cid:
                continue
            try:
                diff_bytes = await ipfs_client.get_bytes(diff_cid)
                weight_diff_str = diff_bytes.decode("utf-8")
                client_updates.append(
                    {"client_id": client_id, "weight_diff": weight_diff_str}
                )
            except Exception as e:
                logger.warning(
                    f"Diagnosis: could not fetch diff for client {client_id}: {e}"
                )

    if not client_updates:
        return []

    # Load test dataset and run regression diagnosis (sync)
    try:
        dataset = get_dataset()
        test_dataset = dataset.load_test_data()
        test_loader = DataLoader(
            test_dataset, batch_size=64, shuffle=False, num_workers=0
        )
    except Exception as e:
        logger.warning(f"Diagnosis: could not load test data: {e}")
        return []

    diagnosis = RegressionDiagnosis()
    problematic = diagnosis.diagnose_regression_leave_one_out(
        previous_weights,
        client_updates,
        test_loader,
        baseline_accuracy=baseline_accuracy,
        accuracy_threshold=accuracy_threshold,
    )
    if problematic:
        logger.info(f"Diagnosis identified problematic client(s): {problematic}")
    return problematic


async def identify_unreliable_clients(
    blockchain_client: FabricClient,
    bad_version_id: str,
    good_version_id: str,
    baseline_accuracy: float = 0.0,
    accuracy_threshold: float = 0.5,
) -> List[str]:
    """
    Identify client IDs that caused regression by testing each client's diff individually.

    Uses leave-one-out: fetches each contributing client's weight diff from provenance
    (client_weight_diff_cids), applies it alone to the good (parent) weights, validates;
    clients that cause a significant accuracy drop are flagged. Avoids incorrectly
    excluding good clients (e.g. delta-based "new contributors" heuristic is not used).

    Args:
        blockchain_client: Fabric client
        bad_version_id: Version that failed validation (accuracy drop)
        good_version_id: Last known good version (e.g. best checkpoint)
        baseline_accuracy: Good version accuracy (threshold for regression)
        accuracy_threshold: Min accuracy drop to consider regression

    Returns:
        List of client IDs to exclude after rollback (may be empty if no per-client CIDs or diagnosis fails)
    """
    try:
        bad_provenance = await blockchain_client.get_model_provenance(bad_version_id)
        good_provenance = await blockchain_client.get_model_provenance(good_version_id)
    except Exception as e:
        logger.warning(
            f"Could not fetch provenance for unreliable client identification: {e}. "
            "Skipping exclusion."
        )
        return []

    logger.info(
        "Identifying unreliable clients by testing each diff individually (leave-one-out)."
    )
    return await _diagnose_by_individual_diffs(
        bad_provenance,
        good_provenance,
        baseline_accuracy=baseline_accuracy,
        accuracy_threshold=accuracy_threshold,
    )


async def find_rollback_target_without_clients(
    blockchain_client: FabricClient,
    start_version_id: str,
    exclude_client_ids: List[str],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Find the latest model version (walking chain backwards from start) that does
    not have any of exclude_client_ids in its contributing client_ids.

    Rollback should be to the last model that didn't have the unreliable client
    in provenance.

    Args:
        blockchain_client: Fabric client
        start_version_id: Version to start walking from (e.g. bad version's parent)
        exclude_client_ids: Client IDs that must not appear in the chosen version

    Returns:
        (target_version_id, target_weights_cid) or (None, None) if not found
    """
    if not exclude_client_ids:
        return None, None

    exclude_set = set(exclude_client_ids)
    current_id: Optional[str] = start_version_id

    while current_id:
        try:
            provenance = await blockchain_client.get_model_provenance(current_id)
        except Exception as e:
            logger.warning(f"Failed to get provenance for {current_id}: {e}")
            return None, None

        version_client_ids = set(_client_ids_from_provenance(provenance))
        if not (version_client_ids & exclude_set):
            # This version has none of the excluded clients
            cid = _ipfs_cid_from_provenance(provenance)
            if cid:
                logger.info(
                    f"Rollback target (no excluded clients in provenance): {current_id}, CID={cid}"
                )
                return current_id, cid
            logger.warning(
                f"Version {current_id} has no IPFS CID in provenance, skipping"
            )

        current_id = provenance.get("parent_version_id")

    logger.warning(
        "No rollback target found without excluded clients in chain; "
        "use best checkpoint."
    )
    return None, None


async def get_parent_version_and_cid(
    blockchain_client: FabricClient, version_id: str
) -> Tuple[Optional[str], Optional[str]]:
    """Get parent_version_id and ipfs_cid for a version (for starting rollback walk)."""
    try:
        provenance = await blockchain_client.get_model_provenance(version_id)
        parent_id = provenance.get("parent_version_id")
        cid = _ipfs_cid_from_provenance(provenance)
        return parent_id, cid
    except Exception as e:
        logger.warning(f"Failed to get provenance for {version_id}: {e}")
        return None, None
