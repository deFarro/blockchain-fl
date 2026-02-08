// Package chain provides a minimal in-process blockchain for development mode.
// Blocks are linked by hash (PrevHash); no proof-of-work.
package chain

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sync"
	"time"
)

// RollbackBlockVersionID is the VersionID used for rollback-event blocks (not model versions).
const RollbackBlockVersionID = "_rollback_"

// Block is a single block in the local chain.
type Block struct {
	Index     int       `json:"index"`
	PrevHash  string    `json:"prev_hash"`
	Hash      string    `json:"hash"`
	Timestamp int64     `json:"timestamp"`
	VersionID string    `json:"version_id"`
	Payload   []byte    `json:"-"` // serialized model version, not included in hash for stability
}

// PayloadForHash returns the data that is hashed (index, prev, timestamp, version_id).
// Payload is not hashed so that validation/rollback updates don't change block hash.
func (b *Block) PayloadForHash() []byte {
	return []byte(fmt.Sprintf("%d|%s|%d|%s", b.Index, b.PrevHash, b.Timestamp, b.VersionID))
}

// ComputeHash sets b.Hash from PrevHash, Timestamp, VersionID.
func (b *Block) ComputeHash() {
	h := sha256.Sum256(b.PayloadForHash())
	b.Hash = hex.EncodeToString(h[:])
}

// LocalChain is an in-memory blockchain: ordered blocks linked by hash.
type LocalChain struct {
	mu     sync.RWMutex
	Blocks []*Block              // ordered chain
	ByID   map[string]*Block     // version_id -> block for O(1) lookup
}

// NewLocalChain creates an empty chain with a genesis block.
func NewLocalChain() *LocalChain {
	c := &LocalChain{
		Blocks: make([]*Block, 0),
		ByID:   make(map[string]*Block),
	}
	genesis := &Block{
		Index:     0,
		PrevHash:  "0",
		Timestamp: time.Now().Unix(),
		VersionID: "",
	}
	genesis.ComputeHash()
	c.Blocks = append(c.Blocks, genesis)
	return c
}

// AddBlock appends a new block with the given versionID and optional payload (stored but not in hash).
// Returns the block hash and the new block.
func (c *LocalChain) AddBlock(versionID string, payload []byte) (blockHash string, block *Block) {
	c.mu.Lock()
	defer c.mu.Unlock()

	prev := c.Blocks[len(c.Blocks)-1]
	b := &Block{
		Index:     len(c.Blocks),
		PrevHash:  prev.Hash,
		Timestamp: time.Now().Unix(),
		VersionID: versionID,
		Payload:   payload,
	}
	b.ComputeHash()
	c.Blocks = append(c.Blocks, b)
	if versionID != RollbackBlockVersionID {
		c.ByID[versionID] = b
	}
	return b.Hash, b
}

// AddRollbackBlock appends a block that stores a rollback event (VersionID = RollbackBlockVersionID).
// Not added to ByID. Returns block hash and the new block.
func (c *LocalChain) AddRollbackBlock(payload []byte) (blockHash string, block *Block) {
	return c.AddBlock(RollbackBlockVersionID, payload)
}

// GetMostRecentRollbackPayload returns the payload of the most recent rollback block (last in chain), or nil.
func (c *LocalChain) GetMostRecentRollbackPayload() []byte {
	c.mu.RLock()
	defer c.mu.RUnlock()
	for i := len(c.Blocks) - 1; i >= 0; i-- {
		if c.Blocks[i].VersionID == RollbackBlockVersionID {
			return c.Blocks[i].Payload
		}
	}
	return nil
}

// BlockByVersionID returns the block for the given version ID, or nil.
func (c *LocalChain) BlockByVersionID(versionID string) *Block {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.ByID[versionID]
}

// Chain returns a copy of the block slice (ordered).
func (c *LocalChain) Chain() []*Block {
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := make([]*Block, len(c.Blocks))
	copy(out, c.Blocks)
	return out
}

// Validate walks the chain and verifies each block's hash and PrevHash link.
func (c *LocalChain) Validate() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if len(c.Blocks) == 0 {
		return false
	}
	prevHash := "0"
	for _, b := range c.Blocks {
		if b.PrevHash != prevHash {
			return false
		}
		expected := *b
		expected.ComputeHash()
		if expected.Hash != b.Hash {
			return false
		}
		prevHash = b.Hash
	}
	return true
}

// Len returns the number of blocks (including genesis).
func (c *LocalChain) Len() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.Blocks)
}

