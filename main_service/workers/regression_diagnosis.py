"""Regression diagnosis to identify problematic clients."""

from typing import Dict, Any, List, Optional, Tuple
import torch
from shared.logger import setup_logger
from shared.models.model import SimpleCNN
from shared.config import settings

logger = setup_logger(__name__)


class RegressionDiagnosis:
    """Diagnoses which client's diff caused model regression."""

    def __init__(self, model: Optional[SimpleCNN] = None):
        """
        Initialize regression diagnosis.

        Args:
            model: Model instance (creates new one if None)
        """
        self.model = model or SimpleCNN(num_classes=10)

    def test_single_client_diff(
        self,
        previous_weights: Dict[str, Any],
        client_diff: Dict[str, Any],
        test_loader: Any,  # DataLoader type
    ) -> Tuple[float, Dict[str, float]]:
        """
        Test a single client's diff by applying it and validating.

        Args:
            previous_weights: Previous model weights
            client_diff: Single client's weight diff
            test_loader: DataLoader for test dataset

        Returns:
            Tuple of (accuracy, metrics_dict)
        """
        # Load previous weights
        self.model.set_weights(previous_weights)

        # Apply single client diff
        self.model.apply_weight_diff(client_diff)

        # Validate on test dataset
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        import torch.nn.functional as F

        with torch.no_grad():
            for data, target in test_loader:
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                loss = F.nll_loss(output, target, reduction="sum")
                total_loss += loss.item()

        accuracy = 100.0 * correct / total
        avg_loss = total_loss / total

        metrics = {"accuracy": accuracy, "loss": avg_loss, "samples": total}

        return accuracy, metrics

    def diagnose_regression(
        self,
        previous_weights: Dict[str, Any],
        client_updates: List[Dict[str, Any]],
        test_loader: Any,
        baseline_accuracy: float,
        accuracy_threshold: float = 0.5,  # Minimum accuracy drop to consider regression
    ) -> List[str]:
        """
        Diagnose which client(s) caused regression by testing each diff individually.

        Args:
            previous_weights: Previous model weights (before problematic iteration)
            client_updates: List of client updates from the problematic iteration
            test_loader: DataLoader for test dataset
            baseline_accuracy: Baseline accuracy to compare against
            accuracy_threshold: Minimum accuracy drop to consider as regression

        Returns:
            List of client IDs that caused regression
        """
        problematic_clients: List[str] = []

        logger.info(
            f"Starting regression diagnosis for {len(client_updates)} clients. "
            f"Baseline accuracy: {baseline_accuracy:.2f}%"
        )

        for update in client_updates:
            client_id = update.get("client_id", "unknown")
            weight_diff_str = update.get("weight_diff", "")

            if not weight_diff_str:
                logger.warning(f"Client {client_id} has no weight diff, skipping")
                continue

            # Deserialize diff
            import json

            try:
                diff_dict = json.loads(weight_diff_str)
                client_diff = {}
                for name, tensor_list in diff_dict.items():
                    client_diff[name] = torch.tensor(tensor_list)
            except Exception as e:
                logger.error(f"Failed to deserialize diff from {client_id}: {str(e)}")
                continue

            # Test this client's diff
            logger.info(f"Testing diff from {client_id}...")
            accuracy, metrics = self.test_single_client_diff(
                previous_weights, client_diff, test_loader
            )

            accuracy_drop = baseline_accuracy - accuracy

            logger.info(
                f"Client {client_id}: accuracy={accuracy:.2f}%, "
                f"drop={accuracy_drop:.2f}% (baseline: {baseline_accuracy:.2f}%)"
            )

            # If accuracy dropped significantly, mark as problematic
            if accuracy_drop >= accuracy_threshold:
                problematic_clients.append(client_id)
                logger.warning(
                    f"Client {client_id} caused regression: "
                    f"accuracy dropped by {accuracy_drop:.2f}%"
                )

        if problematic_clients:
            logger.warning(
                f"Identified {len(problematic_clients)} problematic client(s): "
                f"{problematic_clients}"
            )
        else:
            logger.info(
                "No individual client caused significant regression. "
                "Regression might be due to combination of updates or other factors."
            )

        return problematic_clients

    def test_client_combinations(
        self,
        previous_weights: Dict[str, Any],
        client_updates: List[Dict[str, Any]],
        test_loader: Any,
        baseline_accuracy: float,
        accuracy_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Test different combinations of client diffs to find problematic subset.

        This is a more sophisticated approach that tests combinations:
        - Test each client individually
        - Test pairs of clients
        - Test all combinations to find minimal problematic set

        Args:
            previous_weights: Previous model weights
            client_updates: List of client updates
            test_loader: DataLoader for test dataset
            baseline_accuracy: Baseline accuracy
            accuracy_threshold: Minimum accuracy drop to consider regression

        Returns:
            Dictionary with diagnosis results:
            - problematic_clients: List of client IDs that cause regression
            - test_results: Detailed test results for each combination
        """
        # For now, implement simple individual testing
        # Future: Implement combination testing (pairs, triplets, etc.)
        problematic_clients = self.diagnose_regression(
            previous_weights,
            client_updates,
            test_loader,
            baseline_accuracy,
            accuracy_threshold,
        )

        return {
            "problematic_clients": problematic_clients,
            "test_method": "individual",  # or "combinations" for future enhancement
            "baseline_accuracy": baseline_accuracy,
            "total_clients_tested": len(client_updates),
        }
