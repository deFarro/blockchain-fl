package fabric

import (
	"fmt"
	"log"
	"os"
)

// Note: Fabric SDK is not included to avoid dependency issues.
// When actually using Fabric network, uncomment the SDK imports and
// replace the stub implementations below with actual Fabric SDK calls.

// FabricClient wraps the Hyperledger Fabric Gateway client
// In development mode, this is a stub that always returns "not configured" errors
type FabricClient struct {
	initialized bool
}

// NewFabricClient creates a new Fabric client
func NewFabricClient() *FabricClient {
	return &FabricClient{
		initialized: false,
	}
}

// Initialize connects to the Fabric network
// In development mode (without Fabric SDK), this always returns an error
// which causes the service to fall back to in-memory storage
func (fc *FabricClient) Initialize() error {
	// Check if Fabric network profile is configured
	networkProfile := os.Getenv("FABRIC_NETWORK_PROFILE")
	if networkProfile == "" {
		log.Println("FABRIC_NETWORK_PROFILE not set, using development mode (in-memory storage)")
		return fmt.Errorf("FABRIC_NETWORK_PROFILE not configured")
	}

	// Check if wallet path is configured
	walletPath := os.Getenv("FABRIC_WALLET_PATH")
	if walletPath == "" {
		log.Println("FABRIC_WALLET_PATH not set, using development mode (in-memory storage)")
		return fmt.Errorf("FABRIC_WALLET_PATH not configured")
	}

	// In development mode without Fabric SDK, always return error
	// This causes the service to use in-memory storage
	log.Println("Fabric SDK not available, using development mode (in-memory storage)")
	return fmt.Errorf("Fabric SDK not available - install fabric-sdk-go to use blockchain mode")
}

// IsInitialized returns whether the Fabric client is initialized
func (fc *FabricClient) IsInitialized() bool {
	return fc.initialized
}

// RegisterModelUpdate invokes the chaincode to register a model update
func (fc *FabricClient) RegisterModelUpdate(versionID, parentVersionID, hash, diffHash, ipfsCID, metadataJSON string, iteration, numClients int, clientIDsJSON string) (string, error) {
	if !fc.initialized {
		return "", fmt.Errorf("Fabric client not initialized")
	}

	// Stub implementation - would use Fabric SDK here
	return "", fmt.Errorf("Fabric SDK not available")
}

// RecordValidation invokes the chaincode to record validation results
func (fc *FabricClient) RecordValidation(versionID string, accuracy float64, metricsJSON string) (string, error) {
	if !fc.initialized {
		return "", fmt.Errorf("Fabric client not initialized")
	}

	// Stub implementation - would use Fabric SDK here
	return "", fmt.Errorf("Fabric SDK not available")
}

// RollbackModel invokes the chaincode to record a rollback event
func (fc *FabricClient) RollbackModel(fromVersionID, toVersionID, reason, triggeredBy string) (string, error) {
	if !fc.initialized {
		return "", fmt.Errorf("Fabric client not initialized")
	}

	// Stub implementation - would use Fabric SDK here
	return "", fmt.Errorf("Fabric SDK not available")
}

// GetModelProvenance queries the chaincode to get provenance information
func (fc *FabricClient) GetModelProvenance(versionID string) ([]byte, error) {
	if !fc.initialized {
		return nil, fmt.Errorf("Fabric client not initialized")
	}

	// Stub implementation - would use Fabric SDK here
	return nil, fmt.Errorf("Fabric SDK not available")
}

// Close closes the gateway connection
func (fc *FabricClient) Close() {
	fc.initialized = false
}
