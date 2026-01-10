//go:build !test
// +build !test

// This file is the chaincode for Hyperledger Fabric.
// It requires fabric-contract-api-go dependency and is only built when deploying to Fabric.
// To use: go get github.com/hyperledger/fabric-contract-api-go@v1.2.1

package main

import (
	"encoding/json"
	"fmt"

	"github.com/hyperledger/fabric-contract-api-go/contractapi"
)

// ModelProvenanceContract provides functions for managing model provenance
type ModelProvenanceContract struct {
	contractapi.Contract
}

// ModelVersion represents a model version record stored on blockchain
type ModelVersion struct {
	VersionID         string                 `json:"version_id"`
	ParentVersionID   string                 `json:"parent_version_id,omitempty"`
	Hash              string                 `json:"hash"`
	DiffHash          string                 `json:"diff_hash"`
	IPFSCID           string                 `json:"ipfs_cid"`
	Metadata          map[string]interface{} `json:"metadata"`
	Iteration         int                    `json:"iteration"`
	NumClients        int                    `json:"num_clients"`
	ClientIDs         []string               `json:"client_ids"`
	Timestamp         string                 `json:"timestamp"`
	ValidationStatus  string                 `json:"validation_status"` // pending, passed, failed
	ValidationMetrics map[string]float64     `json:"validation_metrics,omitempty"`
}

// ValidationRecord represents validation results
type ValidationRecord struct {
	VersionID string             `json:"version_id"`
	Accuracy  float64            `json:"accuracy"`
	Metrics   map[string]float64 `json:"metrics"`
	Timestamp string             `json:"timestamp"`
}

// RollbackEvent represents a rollback event
type RollbackEvent struct {
	FromVersionID string `json:"from_version_id"`
	ToVersionID   string `json:"to_version_id"`
	Reason        string `json:"reason"`
	TriggeredBy   string `json:"triggered_by"` // validator_id, user_id, or "automatic"
	Timestamp     string `json:"timestamp"`
	Type          string `json:"type"` // "automatic" or "manual"
}

// RegisterModelUpdate records a new model version on the blockchain
func (s *ModelProvenanceContract) RegisterModelUpdate(ctx contractapi.TransactionContextInterface, versionID string, parentVersionID string, hash string, diffHash string, ipfsCID string, metadataJSON string, iteration int, numClients int, clientIDsJSON string) error {
	// Parse metadata
	var metadata map[string]interface{}
	if err := json.Unmarshal([]byte(metadataJSON), &metadata); err != nil {
		return fmt.Errorf("failed to parse metadata: %v", err)
	}

	// Parse client IDs
	var clientIDs []string
	if err := json.Unmarshal([]byte(clientIDsJSON), &clientIDs); err != nil {
		return fmt.Errorf("failed to parse client IDs: %v", err)
	}

	// Get transaction timestamp
	timestamp, err := ctx.GetStub().GetTxTimestamp()
	if err != nil {
		return fmt.Errorf("failed to get timestamp: %v", err)
	}

	// Create model version
	version := ModelVersion{
		VersionID:        versionID,
		ParentVersionID:  parentVersionID,
		Hash:             hash,
		DiffHash:         diffHash,
		IPFSCID:          ipfsCID,
		Metadata:         metadata,
		Iteration:        iteration,
		NumClients:       numClients,
		ClientIDs:        clientIDs,
		Timestamp:        fmt.Sprintf("%d", timestamp.GetSeconds()),
		ValidationStatus: "pending",
	}

	// Convert to JSON
	versionJSON, err := json.Marshal(version)
	if err != nil {
		return fmt.Errorf("failed to marshal version: %v", err)
	}

	// Store in blockchain state
	return ctx.GetStub().PutState(versionID, versionJSON)
}

// RecordValidation records validation results for a model version
func (s *ModelProvenanceContract) RecordValidation(ctx contractapi.TransactionContextInterface, versionID string, accuracy float64, metricsJSON string) error {
	// Get existing version
	versionJSON, err := ctx.GetStub().GetState(versionID)
	if err != nil {
		return fmt.Errorf("failed to read version: %v", err)
	}
	if versionJSON == nil {
		return fmt.Errorf("version %s does not exist", versionID)
	}

	// Parse existing version
	var version ModelVersion
	if err := json.Unmarshal(versionJSON, &version); err != nil {
		return fmt.Errorf("failed to unmarshal version: %v", err)
	}

	// Parse metrics
	var metrics map[string]float64
	if err := json.Unmarshal([]byte(metricsJSON), &metrics); err != nil {
		return fmt.Errorf("failed to parse metrics: %v", err)
	}

	// Update validation status and metrics
	version.ValidationStatus = "passed"
	if accuracy < 0.5 { // Example threshold, can be configurable
		version.ValidationStatus = "failed"
	}
	version.ValidationMetrics = metrics

	// Get transaction timestamp
	timestamp, err := ctx.GetStub().GetTxTimestamp()
	if err != nil {
		return fmt.Errorf("failed to get timestamp: %v", err)
	}

	// Create validation record
	validationRecord := ValidationRecord{
		VersionID: versionID,
		Accuracy:  accuracy,
		Metrics:   metrics,
		Timestamp: fmt.Sprintf("%d", timestamp.GetSeconds()),
	}

	// Store validation record (with composite key)
	validationKey := fmt.Sprintf("validation_%s", versionID)
	validationJSON, err := json.Marshal(validationRecord)
	if err != nil {
		return fmt.Errorf("failed to marshal validation record: %v", err)
	}
	if err := ctx.GetStub().PutState(validationKey, validationJSON); err != nil {
		return fmt.Errorf("failed to store validation record: %v", err)
	}

	// Update version
	updatedVersionJSON, err := json.Marshal(version)
	if err != nil {
		return fmt.Errorf("failed to marshal updated version: %v", err)
	}

	return ctx.GetStub().PutState(versionID, updatedVersionJSON)
}

// RollbackModel records a rollback event
func (s *ModelProvenanceContract) RollbackModel(ctx contractapi.TransactionContextInterface, fromVersionID string, toVersionID string, reason string, triggeredBy string) error {
	// Verify target version exists
	targetVersionJSON, err := ctx.GetStub().GetState(toVersionID)
	if err != nil {
		return fmt.Errorf("failed to read target version: %v", err)
	}
	if targetVersionJSON == nil {
		return fmt.Errorf("target version %s does not exist", toVersionID)
	}

	// Get transaction timestamp
	timestamp, err := ctx.GetStub().GetTxTimestamp()
	if err != nil {
		return fmt.Errorf("failed to get timestamp: %v", err)
	}

	// Determine rollback type
	rollbackType := "manual"
	if triggeredBy == "automatic" {
		rollbackType = "automatic"
	}

	// Create rollback event
	rollbackEvent := RollbackEvent{
		FromVersionID: fromVersionID,
		ToVersionID:   toVersionID,
		Reason:        reason,
		TriggeredBy:   triggeredBy,
		Timestamp:     fmt.Sprintf("%d", timestamp.GetSeconds()),
		Type:          rollbackType,
	}

	// Store rollback event (with composite key)
	rollbackKey := fmt.Sprintf("rollback_%s_%s", fromVersionID, toVersionID)
	rollbackJSON, err := json.Marshal(rollbackEvent)
	if err != nil {
		return fmt.Errorf("failed to marshal rollback event: %v", err)
	}

	return ctx.GetStub().PutState(rollbackKey, rollbackJSON)
}

// GetModelProvenance retrieves provenance information for a model version
func (s *ModelProvenanceContract) GetModelProvenance(ctx contractapi.TransactionContextInterface, versionID string) (*ModelVersion, error) {
	versionJSON, err := ctx.GetStub().GetState(versionID)
	if err != nil {
		return nil, fmt.Errorf("failed to read version: %v", err)
	}
	if versionJSON == nil {
		return nil, fmt.Errorf("version %s does not exist", versionID)
	}

	var version ModelVersion
	if err := json.Unmarshal(versionJSON, &version); err != nil {
		return nil, fmt.Errorf("failed to unmarshal version: %v", err)
	}

	return &version, nil
}

// VerifyIntegrity verifies the integrity of a model version by comparing hashes
func (s *ModelProvenanceContract) VerifyIntegrity(ctx contractapi.TransactionContextInterface, versionID string, providedHash string) (bool, error) {
	version, err := s.GetModelProvenance(ctx, versionID)
	if err != nil {
		return false, err
	}

	return version.Hash == providedHash, nil
}

// GetValidationHistory retrieves validation history for a model version
func (s *ModelProvenanceContract) GetValidationHistory(ctx contractapi.TransactionContextInterface, versionID string) (*ValidationRecord, error) {
	validationKey := fmt.Sprintf("validation_%s", versionID)
	validationJSON, err := ctx.GetStub().GetState(validationKey)
	if err != nil {
		return nil, fmt.Errorf("failed to read validation record: %v", err)
	}
	if validationJSON == nil {
		return nil, fmt.Errorf("validation record for version %s does not exist", versionID)
	}

	var validationRecord ValidationRecord
	if err := json.Unmarshal(validationJSON, &validationRecord); err != nil {
		return nil, fmt.Errorf("failed to unmarshal validation record: %v", err)
	}

	return &validationRecord, nil
}

func main() {
	chaincode, err := contractapi.NewChaincode(&ModelProvenanceContract{})
	if err != nil {
		fmt.Printf("Error creating model provenance chaincode: %v", err)
		return
	}

	if err := chaincode.Start(); err != nil {
		fmt.Printf("Error starting model provenance chaincode: %v", err)
	}
}
