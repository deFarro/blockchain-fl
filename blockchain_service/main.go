package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/blockchain-fl/blockchain-service/fabric"
	"github.com/gorilla/mux"
)

// ModelVersion represents a model version record
type ModelVersion struct {
	VersionID         string                 `json:"version_id"`
	ParentVersionID   string                 `json:"parent_version_id,omitempty"`
	Hash              string                 `json:"hash"`
	Metadata          map[string]interface{} `json:"metadata"`
	Timestamp         string                 `json:"timestamp"`
	ValidationStatus  string                 `json:"validation_status,omitempty"`
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
	TargetVersionID string `json:"target_version_id"` // Alias for backward compatibility
	Reason        string `json:"reason"`
	TriggeredBy  string `json:"triggered_by"`
	Timestamp    string `json:"timestamp"`
	Type         string `json:"type"`
}

// BlockchainService handles blockchain operations
type BlockchainService struct {
	fabricClient *fabric.FabricClient
	// In-memory storage for development mode (when Fabric is not configured)
	records         map[string]ModelVersion
	rollbackEvents  []RollbackEvent // Track rollback events in development mode
	useFabric       bool
}

// NewBlockchainService creates a new blockchain service
func NewBlockchainService() *BlockchainService {
	service := &BlockchainService{
		records:        make(map[string]ModelVersion),
		rollbackEvents: make([]RollbackEvent, 0),
		useFabric:      false,
	}

	// Try to initialize Fabric client
	fabricClient := fabric.NewFabricClient()
	if err := fabricClient.Initialize(); err == nil {
		service.fabricClient = fabricClient
		service.useFabric = true
		log.Println("Using Hyperledger Fabric for blockchain operations")
	} else {
		log.Println("Fabric not configured, using development mode (in-memory storage)")
		log.Printf("Fabric initialization error: %v", err)
	}

	return service
}

// RegisterModelUpdateRequest represents a request to register a model update
type RegisterModelUpdateRequest struct {
	ModelVersionID  string                 `json:"model_version_id"`
	ParentVersionID string                 `json:"parent_version_id,omitempty"`
	Hash            string                 `json:"hash"`
	Metadata        map[string]interface{} `json:"metadata"`
}

// RegisterModelUpdateResponse represents the response
type RegisterModelUpdateResponse struct {
	TransactionID string `json:"transaction_id"`
	Status        string `json:"status"`
}

// RecordValidationRequest represents a validation request
type RecordValidationRequest struct {
	ModelVersionID string             `json:"model_version_id"`
	Accuracy       float64            `json:"accuracy"`
	Metrics        map[string]float64 `json:"metrics"`
	IPFSCID        string             `json:"ipfs_cid,omitempty"`
}

// RecordValidationResponse represents the response
type RecordValidationResponse struct {
	TransactionID string `json:"transaction_id"`
	Status        string `json:"status"`
}

// RollbackModelRequest represents a rollback request
type RollbackModelRequest struct {
	TargetVersionID string `json:"target_version_id"`
	Reason          string `json:"reason"`
}

// RollbackModelResponse represents the response
type RollbackModelResponse struct {
	TransactionID string `json:"transaction_id"`
	Status        string `json:"status"`
}

// GetMostRecentRollbackResponse represents the response for getting most recent rollback
type GetMostRecentRollbackResponse struct {
	RollbackEvent *RollbackEvent `json:"rollback_event,omitempty"`
	Status        string          `json:"status"`
}

// GetProvenanceResponse represents provenance information
type GetProvenanceResponse struct {
	VersionID         string                 `json:"version_id"`
	ParentVersionID   string                 `json:"parent_version_id,omitempty"`
	Hash              string                 `json:"hash"`
	Metadata          map[string]interface{} `json:"metadata"`
	Timestamp         string                 `json:"timestamp"`
	ValidationStatus  string                 `json:"validation_status,omitempty"`
	ValidationMetrics map[string]float64     `json:"validation_metrics,omitempty"`
}

// HealthResponse represents health check response
type HealthResponse struct {
	Status string `json:"status"`
}

// ListModelsResponse represents the response for listing all models
type ListModelsResponse struct {
	Versions []GetProvenanceResponse `json:"versions"`
	Total    int                     `json:"total"`
}

func (bs *BlockchainService) registerModelUpdate(w http.ResponseWriter, r *http.Request) {
	var req RegisterModelUpdateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	var txID string
	var err error

	if bs.useFabric && bs.fabricClient != nil {
		// Use Fabric SDK
		// Extract metadata from request
		metadataJSON, _ := json.Marshal(req.Metadata)

		// Extract iteration and num_clients from metadata if present
		iteration := 0
		numClients := 0
		clientIDs := []string{}
		if iter, ok := req.Metadata["iteration"].(float64); ok {
			iteration = int(iter)
		}
		if num, ok := req.Metadata["num_clients"].(float64); ok {
			numClients = int(num)
		}
		if ids, ok := req.Metadata["client_ids"].([]interface{}); ok {
			for _, id := range ids {
				if str, ok := id.(string); ok {
					clientIDs = append(clientIDs, str)
				}
			}
		}
		clientIDsJSON, _ := json.Marshal(clientIDs)

		// Get diff_hash and ipfs_cid from metadata
		diffHash := ""
		ipfsCID := ""
		if dh, ok := req.Metadata["diff_hash"].(string); ok {
			diffHash = dh
		}
		if cid, ok := req.Metadata["ipfs_cid"].(string); ok {
			ipfsCID = cid
		}

		txID, err = bs.fabricClient.RegisterModelUpdate(
			req.ModelVersionID,
			req.ParentVersionID,
			req.Hash,
			diffHash,
			ipfsCID,
			string(metadataJSON),
			iteration,
			numClients,
			string(clientIDsJSON),
		)
		if err != nil {
			log.Printf("Fabric transaction failed: %v, falling back to in-memory storage", err)
			bs.useFabric = false // Fallback to in-memory
		}
	}

	if !bs.useFabric {
		// Fallback to in-memory storage
		version := ModelVersion{
			VersionID:       req.ModelVersionID,
			ParentVersionID: req.ParentVersionID,
			Hash:            req.Hash,
			Metadata:        req.Metadata,
			Timestamp:       fmt.Sprintf("%d", time.Now().Unix()),
		}
		bs.records[req.ModelVersionID] = version
		txID = fmt.Sprintf("tx_%s", req.ModelVersionID)
	}

	response := RegisterModelUpdateResponse{
		TransactionID: txID,
		Status:        "success",
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (bs *BlockchainService) recordValidation(w http.ResponseWriter, r *http.Request) {
	var req RecordValidationRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	var txID string
	var err error

	if bs.useFabric && bs.fabricClient != nil {
		// Use Fabric SDK
		metricsJSON, _ := json.Marshal(req.Metrics)
		txID, err = bs.fabricClient.RecordValidation(
			req.ModelVersionID,
			req.Accuracy,
			string(metricsJSON),
			req.IPFSCID,
		)
		if err != nil {
			log.Printf("Fabric transaction failed: %v, falling back to in-memory storage", err)
			bs.useFabric = false
		}
	}

	if !bs.useFabric {
		// Fallback to in-memory storage
		// Update the stored version with validation data
		if version, exists := bs.records[req.ModelVersionID]; exists {
			version.ValidationStatus = "passed"
			if req.Accuracy < 0.5 {
				version.ValidationStatus = "failed"
			}
			version.ValidationMetrics = req.Metrics
			// Update IPFS CID in metadata if provided
			if req.IPFSCID != "" {
				if version.Metadata == nil {
					version.Metadata = make(map[string]interface{})
				}
				version.Metadata["ipfs_cid"] = req.IPFSCID
			}
			bs.records[req.ModelVersionID] = version
		}
		log.Printf("Validation recorded: version=%s, accuracy=%.4f", req.ModelVersionID, req.Accuracy)
		txID = fmt.Sprintf("tx_validation_%s", req.ModelVersionID)
	}

	response := RecordValidationResponse{
		TransactionID: txID,
		Status:        "success",
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (bs *BlockchainService) rollbackModel(w http.ResponseWriter, r *http.Request) {
	var req RollbackModelRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	var txID string
	var err error

	if bs.useFabric && bs.fabricClient != nil {
		// Use Fabric SDK
		// Determine current version (would need to track this, for now use empty)
		fromVersionID := ""        // In a real implementation, track current version
		triggeredBy := "automatic" // Could be extracted from request metadata

		txID, err = bs.fabricClient.RollbackModel(
			fromVersionID,
			req.TargetVersionID,
			req.Reason,
			triggeredBy,
		)
		if err != nil {
			log.Printf("Fabric transaction failed: %v, falling back to in-memory storage", err)
			bs.useFabric = false
		}
	}

	if !bs.useFabric {
		// Fallback to in-memory storage
		log.Printf("Rollback requested: target_version=%s, reason=%s", req.TargetVersionID, req.Reason)
		txID = fmt.Sprintf("tx_rollback_%s", req.TargetVersionID)
		
		// Store rollback event in memory for development mode
		rollbackEvent := RollbackEvent{
			FromVersionID:   "",
			ToVersionID:     req.TargetVersionID,
			TargetVersionID: req.TargetVersionID,
			Reason:          req.Reason,
			TriggeredBy:     "manual",
			Timestamp:       fmt.Sprintf("%d", time.Now().Unix()),
			Type:            "manual",
		}
		bs.rollbackEvents = append(bs.rollbackEvents, rollbackEvent)
	}

	response := RollbackModelResponse{
		TransactionID: txID,
		Status:        "success",
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (bs *BlockchainService) getMostRecentRollback(w http.ResponseWriter, r *http.Request) {
	var rollbackEvent *RollbackEvent

	if bs.useFabric && bs.fabricClient != nil {
		// Use Fabric SDK
		rollbackJSON, err := bs.fabricClient.GetMostRecentRollback()
		if err != nil {
			log.Printf("Fabric query failed: %v, falling back to in-memory storage", err)
			bs.useFabric = false
		} else {
			// Parse rollback event
			var event RollbackEvent
			if err := json.Unmarshal(rollbackJSON, &event); err == nil {
				// Set TargetVersionID for backward compatibility
				if event.TargetVersionID == "" && event.ToVersionID != "" {
					event.TargetVersionID = event.ToVersionID
				}
				rollbackEvent = &event
			}
		}
	}

	if !bs.useFabric {
		// Fallback to in-memory storage
		// Return the most recent rollback event from memory
		if len(bs.rollbackEvents) > 0 {
			// Get the most recent rollback event (last one in the slice)
			mostRecent := bs.rollbackEvents[len(bs.rollbackEvents)-1]
			rollbackEvent = &mostRecent
		} else {
			rollbackEvent = nil
		}
	}

	response := GetMostRecentRollbackResponse{
		RollbackEvent: rollbackEvent,
		Status:        "success",
	}

	if rollbackEvent == nil {
		response.Status = "not_found"
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (bs *BlockchainService) getProvenance(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	versionID := vars["version_id"]

	if bs.useFabric && bs.fabricClient != nil {
		// Use Fabric SDK
		provenanceJSON, err := bs.fabricClient.GetModelProvenance(versionID)
		if err == nil {
			// Return the JSON directly from Fabric
			w.Header().Set("Content-Type", "application/json")
			w.Write(provenanceJSON)
			return
		}
		log.Printf("Fabric query failed: %v, falling back to in-memory storage", err)
		bs.useFabric = false
	}

	// Fallback to in-memory storage
	version, exists := bs.records[versionID]
	if !exists {
		http.Error(w, "Version not found", http.StatusNotFound)
		return
	}

	response := GetProvenanceResponse{
		VersionID:         version.VersionID,
		ParentVersionID:   version.ParentVersionID,
		Hash:              version.Hash,
		Metadata:          version.Metadata,
		Timestamp:         version.Timestamp,
		ValidationStatus:  version.ValidationStatus,
		ValidationMetrics: version.ValidationMetrics,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (bs *BlockchainService) listModels(w http.ResponseWriter, r *http.Request) {
	var versions []GetProvenanceResponse

	if bs.useFabric && bs.fabricClient != nil {
		// Use Fabric SDK - would query all states here
		// For now, fall back to in-memory storage
		log.Printf("Fabric list models not implemented, falling back to in-memory storage")
		bs.useFabric = false
	}

	if !bs.useFabric {
		// Fallback to in-memory storage
		for _, version := range bs.records {
			versions = append(versions, GetProvenanceResponse{
				VersionID:         version.VersionID,
				ParentVersionID:   version.ParentVersionID,
				Hash:              version.Hash,
				Metadata:          version.Metadata,
				Timestamp:         version.Timestamp,
				ValidationStatus:  version.ValidationStatus,
				ValidationMetrics: version.ValidationMetrics,
			})
		}
	}

	response := ListModelsResponse{
		Versions: versions,
		Total:    len(versions),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (bs *BlockchainService) health(w http.ResponseWriter, r *http.Request) {
	response := HealthResponse{
		Status: "healthy",
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func main() {
	port := os.Getenv("BLOCKCHAIN_SERVICE_PORT")
	if port == "" {
		port = "8080"
	}

	service := NewBlockchainService()

	// Cleanup on exit
	defer func() {
		if service.fabricClient != nil {
			service.fabricClient.Close()
		}
	}()

	r := mux.NewRouter()
	r.HandleFunc("/health", service.health).Methods("GET")
	r.HandleFunc("/api/v1/model/register", service.registerModelUpdate).Methods("POST")
	r.HandleFunc("/api/v1/model/validate", service.recordValidation).Methods("POST")
	r.HandleFunc("/api/v1/model/rollback", service.rollbackModel).Methods("POST")
	r.HandleFunc("/api/v1/model/rollback/latest", service.getMostRecentRollback).Methods("GET")
	r.HandleFunc("/api/v1/model/provenance/{version_id}", service.getProvenance).Methods("GET")
	r.HandleFunc("/api/v1/model/list", service.listModels).Methods("GET")

	log.Printf("Blockchain service starting on port %s", port)
	if service.useFabric {
		log.Println("Connected to Hyperledger Fabric network")
	} else {
		log.Println("⚠ Running in development mode (in-memory storage)")
		log.Println("  To use Fabric, configure FABRIC_NETWORK_PROFILE and FABRIC_WALLET_PATH")
	}
	log.Fatal(http.ListenAndServe(fmt.Sprintf(":%s", port), r))
}
