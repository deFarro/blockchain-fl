package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"runtime"
	"time"

	"github.com/blockchain-fl/blockchain-service/chain"
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

// validationOverlay holds validation results for a version (chain blocks are immutable).
type validationOverlay struct {
	ValidationStatus  string
	ValidationMetrics map[string]float64
	IPFSCID           string
}

// BlockchainService handles blockchain operations
type BlockchainService struct {
	fabricClient *fabric.FabricClient
	useFabric    bool
	// Development mode: local chain only (versions + rollback events on chain)
	localChain        *chain.LocalChain
	validationOverlay  map[string]validationOverlay // version_id -> validation data
}

// NewBlockchainService creates a new blockchain service
func NewBlockchainService() *BlockchainService {
	service := &BlockchainService{
		localChain:        chain.NewLocalChain(),
		validationOverlay: make(map[string]validationOverlay),
		useFabric:         false,
	}

	// Try to initialize Fabric client
	fabricClient := fabric.NewFabricClient()
	if err := fabricClient.Initialize(); err == nil {
		service.fabricClient = fabricClient
		service.useFabric = true
		log.Println("Using Hyperledger Fabric for blockchain operations")
	} else {
		log.Println("Development mode: using local blockchain only (chain-only storage)")
		log.Println("  Verify: GET http://localhost:8080/api/v1/chain returns length and valid")
		log.Printf("  Fabric init skipped: %v", err)
	}

	return service
}

// getVersionFromChain returns the model version for versionID from the local chain, with validation overlay applied. ok is false if not found.
func (bs *BlockchainService) getVersionFromChain(versionID string) (ModelVersion, bool) {
	block := bs.localChain.BlockByVersionID(versionID)
	if block == nil || len(block.Payload) == 0 {
		return ModelVersion{}, false
	}
	var v ModelVersion
	if err := json.Unmarshal(block.Payload, &v); err != nil {
		return ModelVersion{}, false
	}
	if overlay, has := bs.validationOverlay[versionID]; has {
		v.ValidationStatus = overlay.ValidationStatus
		v.ValidationMetrics = overlay.ValidationMetrics
		if overlay.IPFSCID != "" {
			if v.Metadata == nil {
				v.Metadata = make(map[string]interface{})
			}
			v.Metadata["ipfs_cid"] = overlay.IPFSCID
		}
	}
	return v, true
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

// SystemMetricsResponse represents system metrics for the blockchain-service process (memory, etc.)
type SystemMetricsResponse struct {
	Timestamp string              `json:"timestamp"`
	Memory    MemStatsSnapshot    `json:"memory"`
}

// MemStatsSnapshot is a subset of runtime.MemStats for export
type MemStatsSnapshot struct {
	AllocBytes      uint64 `json:"alloc_bytes"`
	TotalAllocBytes uint64 `json:"total_alloc_bytes"`
	SysBytes        uint64 `json:"sys_bytes"`
	NumGC           uint32 `json:"num_gc"`
}

// HealthResponse represents health check response
type HealthResponse struct {
	Status string `json:"status"`
}

// ChainInfoResponse is returned by GET /api/v1/chain (dev mode)
type ChainInfoResponse struct {
	Mode    string `json:"mode"`    // "local_chain" when using chain-only storage
	Length  int    `json:"length"`  // number of blocks including genesis
	Valid   bool   `json:"valid"`   // chain integrity (hash links)
	Message string `json:"message"` // how to interpret the response
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
		// Chain-only: append block with version payload
		version := ModelVersion{
			VersionID:       req.ModelVersionID,
			ParentVersionID: req.ParentVersionID,
			Hash:            req.Hash,
			Metadata:        req.Metadata,
			Timestamp:       fmt.Sprintf("%d", time.Now().Unix()),
		}
		payload, _ := json.Marshal(version)
		blockHash, _ := bs.localChain.AddBlock(req.ModelVersionID, payload)
		txID = blockHash
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
		// Chain-only: validation overlay (block must exist in chain)
		if bs.localChain.BlockByVersionID(req.ModelVersionID) == nil {
			http.Error(w, "version not found in chain", http.StatusNotFound)
			return
		}
		status := "passed"
		if req.Accuracy < 0.5 {
			status = "failed"
		}
		bs.validationOverlay[req.ModelVersionID] = validationOverlay{
			ValidationStatus:  status,
			ValidationMetrics: req.Metrics,
			IPFSCID:           req.IPFSCID,
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
		// Chain-only: append rollback event as a block
		rollbackEvent := RollbackEvent{
			FromVersionID:   "",
			ToVersionID:     req.TargetVersionID,
			TargetVersionID: req.TargetVersionID,
			Reason:          req.Reason,
			TriggeredBy:     "manual",
			Timestamp:       fmt.Sprintf("%d", time.Now().Unix()),
			Type:            "manual",
		}
		payload, _ := json.Marshal(rollbackEvent)
		txID, _ = bs.localChain.AddRollbackBlock(payload)
		log.Printf("Rollback requested: target_version=%s, reason=%s", req.TargetVersionID, req.Reason)
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
		// Chain-only: most recent rollback block payload
		payload := bs.localChain.GetMostRecentRollbackPayload()
		if len(payload) > 0 {
			var event RollbackEvent
			if err := json.Unmarshal(payload, &event); err == nil {
				if event.TargetVersionID == "" && event.ToVersionID != "" {
					event.TargetVersionID = event.ToVersionID
				}
				rollbackEvent = &event
			}
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

	// Chain-only: read from chain + overlay
	version, exists := bs.getVersionFromChain(versionID)
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
		// Chain-only: iterate blocks (skip genesis and rollback blocks), merge overlay
		for _, block := range bs.localChain.Chain() {
			if block.VersionID == "" || block.VersionID == chain.RollbackBlockVersionID {
				continue
			}
			version, ok := bs.getVersionFromChain(block.VersionID)
			if !ok {
				continue
			}
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

func (bs *BlockchainService) chainInfo(w http.ResponseWriter, r *http.Request) {
	if bs.useFabric {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(ChainInfoResponse{
			Mode: "fabric", Length: 0, Valid: true,
			Message: "Using Hyperledger Fabric; chain info not exposed here.",
		})
		return
	}
	response := ChainInfoResponse{
		Mode:    "local_chain",
		Length:  bs.localChain.Len(),
		Valid:   bs.localChain.Validate(),
		Message: "Development mode: storage is chain-only. length=blocks (genesis+versions), valid=hash chain integrity.",
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (bs *BlockchainService) systemMetrics(w http.ResponseWriter, r *http.Request) {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	response := SystemMetricsResponse{
		Timestamp: time.Now().UTC().Format(time.RFC3339),
		Memory: MemStatsSnapshot{
			AllocBytes:      m.Alloc,
			TotalAllocBytes: m.TotalAlloc,
			SysBytes:        m.Sys,
			NumGC:           m.NumGC,
		},
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
	r.HandleFunc("/api/v1/system-metrics", service.systemMetrics).Methods("GET")
	r.HandleFunc("/api/v1/model/register", service.registerModelUpdate).Methods("POST")
	r.HandleFunc("/api/v1/model/validate", service.recordValidation).Methods("POST")
	r.HandleFunc("/api/v1/model/rollback", service.rollbackModel).Methods("POST")
	r.HandleFunc("/api/v1/model/rollback/latest", service.getMostRecentRollback).Methods("GET")
	r.HandleFunc("/api/v1/model/provenance/{version_id}", service.getProvenance).Methods("GET")
	r.HandleFunc("/api/v1/model/list", service.listModels).Methods("GET")
	r.HandleFunc("/api/v1/chain", service.chainInfo).Methods("GET")

	log.Printf("Blockchain service starting on port %s", port)
	if service.useFabric {
		log.Println("Connected to Hyperledger Fabric network")
	} else {
		log.Println("Development mode: local blockchain only (chain-only storage)")
		log.Printf("  Verify chain: GET http://localhost:%s/api/v1/chain → mode=local_chain, length, valid", port)
		log.Println("  To use Fabric, set FABRIC_NETWORK_PROFILE and FABRIC_WALLET_PATH")
	}
	log.Fatal(http.ListenAndServe(fmt.Sprintf(":%s", port), r))
}
