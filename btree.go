// Copyright 2014 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package btree implements in-memory B-Trees of arbitrary degree.
//
// btree implements an in-memory B-Tree for use as an ordered data structure.
// It is not meant for persistent storage solutions.
//
// It has a flatter structure than an equivalent red-black or other binary tree,
// which in some cases yields better memory usage and/or performance.
// See some discussion on the matter here:
//   http://google-opensource.blogspot.com/2013/01/c-containers-that-save-memory-and-time.html
// Note, though, that this project is in no way related to the C++ B-Tree
// implmentation written about there.
//
// Within this tree, each node contains a slice of items and a (possibly nil)
// slice of children.  For basic numeric values or raw structs, this can cause
// efficiency differences when compared to equivalent C++ template code that
// stores values in arrays within the node:
//   * Due to the overhead of storing values as interfaces (each
//     value needs to be stored as the value itself, then 2 words for the
//     interface pointing to that value and its type), resulting in higher
//     memory use.
//   * Since interfaces can point to values anywhere in memory, values are
//     most likely not stored in contiguous blocks, resulting in a higher
//     number of cache misses.
// These issues don't tend to matter, though, when working with strings or other
// heap-allocated structures, since C++-equivalent structures also must store
// pointers and also distribute their values across the heap.
//
// This implementation is designed to be a drop-in replacement to gollrb.LLRB
// trees, (http://github.com/petar/gollrb), an excellent and probably the most
// widely used ordered tree implementation in the Go ecosystem currently.
// Its functions, therefore, exactly mirror those of
// llrb.LLRB where possible.  Unlike gollrb, though, we currently don't
// support storing multiple equivalent values or backwards iteration.
package btree

import (
	"fmt"
	"io"
	"sort"
	"strings"
)

// Item represents a single object in the tree.
type Item interface {
	// Less tests whether the current item is less than the given argument.
	//
	// This must provide a strict weak ordering.
	// If !a.Less(b) && !b.Less(a), we treat this to mean a == b (i.e. we can only
	// hold one of either a or b in the tree).
	Less(than Item) bool
}

const (
	DefaultFreeListSize = 32
)

// FreeList represents a free list of btree nodes. By default each
// BTree has its own FreeList, but multiple BTrees can share the same
// FreeList.
// Two Btrees using the same freelist are not safe for concurrent write access.
type FreeList struct {
	freelist []*node
}

// NewFreeList creates a new free list.
// size is the maximum size of the returned free list.
func NewFreeList(size int) *FreeList {
	return &FreeList{freelist: make([]*node, 0, size)}
}

func (f *FreeList) newNode() (n *node) {
	index := len(f.freelist) - 1
	if index < 0 {
		return new(node)
	}
	f.freelist, n = f.freelist[:index], f.freelist[index]
	return
}

func (f *FreeList) freeNode(n *node) {
	if len(f.freelist) < cap(f.freelist) {
		f.freelist = append(f.freelist, n)
	}
}

// ItemIterator allows callers of Ascend* to iterate in-order over portions of
// the tree.  When this function returns false, iteration will stop and the
// associated Ascend* function will immediately return.
type ItemIterator func(i Item) bool

// New creates a new B-Tree with the given degree.
//
// New(2), for example, will create a 2-3-4 tree (each node contains 1-3 items
// and 2-4 children).
func New(degree int) *BTree {
	return NewWithFreeList(degree, NewFreeList(DefaultFreeListSize))
}

// NewWithFreeList creates a new B-Tree that uses the given node free list.
func NewWithFreeList(degree int, f *FreeList) *BTree {
	if degree <= 1 {
		panic("bad degree")
	}
	return &BTree{
		op: &btreeOp{
			degree:   degree,
			freelist: f,
		},
	}
}

// items stores items in a node.
type items []Item

// insertAt inserts a value into the given index, pushing all subsequent values
// forward.
func (s *items) insertAt(index int, item Item) {
	*s = append(*s, nil)
	if index < len(*s) {
		copy((*s)[index+1:], (*s)[index:])
	}
	(*s)[index] = item
}

// removeAt removes a value at a given index, pulling all subsequent values
// back.
func (s *items) removeAt(index int) Item {
	item := (*s)[index]
	copy((*s)[index:], (*s)[index+1:])
	(*s)[len(*s)-1] = nil
	*s = (*s)[:len(*s)-1]
	return item
}

// pop removes and returns the last element in the list.
func (s *items) pop() (out Item) {
	index := len(*s) - 1
	out = (*s)[index]
	(*s)[index] = nil
	*s = (*s)[:index]
	return
}

// find returns the index where the given item should be inserted into this
// list.  'found' is true if the item already exists in the list at the given
// index.
func (s items) find(item Item) (index int, found bool) {
	i := sort.Search(len(s), func(i int) bool {
		return item.Less(s[i])
	})
	if i > 0 && !s[i-1].Less(item) {
		return i - 1, true
	}
	return i, false
}

// children stores child nodes in a node.
type children []*node

// insertAt inserts a value into the given index, pushing all subsequent values
// forward.
func (s *children) insertAt(index int, n *node) {
	*s = append(*s, nil)
	if index < len(*s) {
		copy((*s)[index+1:], (*s)[index:])
	}
	(*s)[index] = n
}

// removeAt removes a value at a given index, pulling all subsequent values
// back.
func (s *children) removeAt(index int) *node {
	n := (*s)[index]
	copy((*s)[index:], (*s)[index+1:])
	(*s)[len(*s)-1] = nil
	*s = (*s)[:len(*s)-1]
	return n
}

// pop removes and returns the last element in the list.
func (s *children) pop() (out *node) {
	index := len(*s) - 1
	out = (*s)[index]
	(*s)[index] = nil
	*s = (*s)[:index]
	return
}

// node is an internal node in a tree.
//
// It must at all times maintain the invariant that either
//   * len(children) == 0, len(items) unconstrained
//   * len(children) == len(items) + 1
type node struct {
	items    items
	children children
	op       *btreeOp
}

// _split is the helper function for both ephemeral and persistent forms
//  of split.
func (n *node) _split(
	i int,
	writables copyOnWriteSet,
	newNode func(*node, copyOnWriteSet) *node) (Item, *node) {
	item := n.items[i]
	next := newNode(n, writables)
	next.items = append(next.items, n.items[i+1:]...)
	n.items = n.items[:i]
	if len(n.children) > 0 {
		next.children = append(next.children, n.children[i+1:]...)
		n.children = n.children[:i+1]
	}
	return item, next
}

// newNodeShim is the ephemeral shim to create a new node
func newNodeShim(n *node, writables copyOnWriteSet) *node {
	return n.op.newNode()
}

// split splits the given node at the given index.  The current node shrinks,
// and this function returns the item that existed at that index and a new node
// containing all items/children after it.
func (n *node) split(i int) (Item, *node) {
	return n._split(i, nil, newNodeShim)
}

// pNewNodeShim is the persistent shim to create a new node
func pNewNodeShim(n *node, writables copyOnWriteSet) *node {
	return writables.newNode(n.op)
}

// pSplit is the persistent version of split
func (n *node) pSplit(i int, writables copyOnWriteSet) (*node, Item, *node) {
	wn := writables.writableNode(n)
	item, next := wn._split(i, writables, pNewNodeShim)
	return wn, item, next
}

// _maybeSplitChild is the helper function for both ephemeral and persistent
// forms of maybeSplitChild.
func (n *node) _maybeSplitChild(
	i, maxItems int,
	writables copyOnWriteSet,
	splitChild func(*node, int, int, copyOnWriteSet) (Item, *node)) bool {
	if len(n.children[i].items) < maxItems {
		return false
	}
	item, second := splitChild(n, i, maxItems/2, writables)
	n.items.insertAt(i, item)
	n.children.insertAt(i+1, second)
	return true
}

// splitChildShim is the ephemeral shim to split a child
func splitChildShim(
	n *node,
	childIndex int,
	splitIndex int,
	writables copyOnWriteSet) (Item, *node) {
	first := n.children[childIndex]
	return first.split(splitIndex)
}

// maybeSplitChild checks if a child should be split, and if so splits it.
// Returns whether or not a split occurred.
func (n *node) maybeSplitChild(i, maxItems int) bool {
	return n._maybeSplitChild(i, maxItems, nil, splitChildShim)
}

// pSplitChildShim is the persistent shim to split a child
func pSplitChildShim(
	n *node,
	childIndex int,
	splitIndex int,
	writables copyOnWriteSet) (Item, *node) {
	child := n.children[childIndex]
	child, item, second := child.pSplit(splitIndex, writables)
	n.children[childIndex] = child
	return item, second
}

// pMaybeSplitChild is the persistent version of maybeSplitChild
// always returning a writable version of this node.
func (n *node) pMaybeSplitChild(
	i, maxItems int, writables copyOnWriteSet) (*node, bool) {
	wn := writables.writableNode(n)
	result := wn._maybeSplitChild(
		i, maxItems, writables, pSplitChildShim)
	return wn, result
}

// _insert is the helper function for both ephemeral and persistent
// forms of insert.
func (n *node) _insert(
	item Item,
	maxItems int,
	writables copyOnWriteSet,
	maybeSplitChild func(*node, int, int, copyOnWriteSet) bool,
	childInsert func(*node, int, Item, int, copyOnWriteSet) Item,
) Item {
	i, found := n.items.find(item)
	if found {
		out := n.items[i]
		n.items[i] = item
		return out
	}
	if len(n.children) == 0 {
		n.items.insertAt(i, item)
		return nil
	}
	if maybeSplitChild(n, i, maxItems, writables) {
		inTree := n.items[i]
		switch {
		case item.Less(inTree):
			// no change, we want first split node
		case inTree.Less(item):
			i++ // we want second split node
		default:
			out := n.items[i]
			n.items[i] = item
			return out
		}
	}
	return childInsert(n, i, item, maxItems, writables)
}

// maybeSplitChildShim is the ephemeral shim to maybe split a child
func maybeSplitChildShim(
	n *node,
	childIndex int,
	maxItems int,
	writables copyOnWriteSet) bool {
	return n.maybeSplitChild(childIndex, maxItems)
}

// childInsertShim is the ephemeral shim to insert into a child
func childInsertShim(
	n *node,
	childIndex int,
	item Item,
	maxItems int,
	writables copyOnWriteSet) Item {
	return n.children[childIndex].insert(item, maxItems)
}

// insert inserts an item into the subtree rooted at this node, making sure
// no nodes in the subtree exceed maxItems items.  Should an equivalent item be
// be found/replaced by insert, it will be returned.
func (n *node) insert(item Item, maxItems int) Item {
	return n._insert(
		item, maxItems, nil, maybeSplitChildShim, childInsertShim)
}

// pMaybeSplitChildShim is the persistent shim to maybe split a child
func pMaybeSplitChildShim(
	n *node,
	childIndex int,
	maxItems int,
	writables copyOnWriteSet) bool {
	_, result := n.pMaybeSplitChild(childIndex, maxItems, writables)
	return result
}

// pChildInsertShim is the persistent shim to insert into a child
func pChildInsertShim(
	n *node,
	childIndex int,
	item Item,
	maxItems int,
	writables copyOnWriteSet) Item {
	newChild, out := n.children[childIndex].pInsert(
		item, maxItems, writables)
	n.children[childIndex] = newChild
	return out
}

// pInsert is the persistent form of insert
func (n *node) pInsert(
	item Item, maxItems int, writables copyOnWriteSet) (*node, Item) {
	wn := writables.writableNode(n)
	result := wn._insert(
		item,
		maxItems,
		writables,
		pMaybeSplitChildShim,
		pChildInsertShim)
	return wn, result
}

// get finds the given key in the subtree and returns it.
func (n *node) get(key Item) Item {
	i, found := n.items.find(key)
	if found {
		return n.items[i]
	} else if len(n.children) > 0 {
		return n.children[i].get(key)
	}
	return nil
}

// min returns the first item in the subtree.
func min(n *node) Item {
	if n == nil {
		return nil
	}
	for len(n.children) > 0 {
		n = n.children[0]
	}
	if len(n.items) == 0 {
		return nil
	}
	return n.items[0]
}

// max returns the last item in the subtree.
func max(n *node) Item {
	if n == nil {
		return nil
	}
	for len(n.children) > 0 {
		n = n.children[len(n.children)-1]
	}
	if len(n.items) == 0 {
		return nil
	}
	return n.items[len(n.items)-1]
}

// toRemove details what item to remove in a node.remove call.
type toRemove int

const (
	removeItem toRemove = iota // removes the given item
	removeMin                  // removes smallest item in the subtree
	removeMax                  // removes largest item in the subtree
)

// _remove is the helper function for both ephemeral and persistent
// forms of remove.
func (n *node) _remove(
	item Item,
	minItems int,
	typ toRemove,
	writables copyOnWriteSet,
	growChildAndRemove func(
		*node, int, Item, int, toRemove, copyOnWriteSet) Item,
	childRemove func(
		*node, int, Item, int, toRemove, copyOnWriteSet) Item,
) Item {
	var i int
	var found bool
	switch typ {
	case removeMax:
		if len(n.children) == 0 {
			return n.items.pop()
		}
		i = len(n.items)
	case removeMin:
		if len(n.children) == 0 {
			return n.items.removeAt(0)
		}
		i = 0
	case removeItem:
		i, found = n.items.find(item)
		if len(n.children) == 0 {
			if found {
				return n.items.removeAt(i)
			}
			return nil
		}
	default:
		panic("invalid type")
	}
	// If we get to here, we have children.
	child := n.children[i]
	if len(child.items) <= minItems {
		return growChildAndRemove(n, i, item, minItems, typ, writables)
	}
	// Either we had enough items to begin with, or we've done some
	// merging/stealing, because we've got enough now and we're ready to return
	// stuff.
	if found {
		// The item exists at index 'i', and the child we've selected can give us a
		// predecessor, since if we've gotten here it's got > minItems items in it.
		// We use our special-case 'remove' call with typ=maxItem to pull the
		// predecessor of item i (the rightmost leaf of our immediate left child)
		// and set it into where we pulled the item from.
		return childRemove(n, i, nil, minItems, removeMax, writables)
	}
	// Final recursive call.  Once we're here, we know that the item isn't in this
	// node and that the child is big enough to remove from.
	return childRemove(n, i, item, minItems, typ, writables)
}

// growChildAndRemoveShim is the ephemeral shim
// for growing a child and removing
func growChildAndRemoveShim(
	n *node,
	childIndex int,
	item Item,
	minItems int,
	typ toRemove,
	writables copyOnWriteSet) Item {
	return n.growChildAndRemove(childIndex, item, minItems, typ)
}

// childRemoveShim is the ephemeral shim for removing from a child
func childRemoveShim(
	n *node,
	childIndex int,
	item Item,
	minItems int,
	typ toRemove,
	writables copyOnWriteSet) Item {
	if item == nil {
		out := n.items[childIndex]
		n.items[childIndex] = n.children[childIndex].remove(
			nil, minItems, removeMax)
		return out
	}
	return n.children[childIndex].remove(item, minItems, typ)
}

// remove removes an item from the subtree rooted at this node.
func (n *node) remove(item Item, minItems int, typ toRemove) Item {
	return n._remove(
		item, minItems, typ, nil, growChildAndRemoveShim, childRemoveShim)
}

// pGrowChildAndRemoveShim is the persistent shim
// for growing a child and removing
func pGrowChildAndRemoveShim(
	n *node,
	childIndex int,
	item Item,
	minItems int,
	typ toRemove,
	writables copyOnWriteSet) Item {
	_, result := n.pGrowChildAndRemove(
		childIndex, item, minItems, typ, writables)
	return result
}

// pChildRemoveShim is the persistent shim for removing from a child
func pChildRemoveShim(
	n *node,
	childIndex int,
	item Item,
	minItems int,
	typ toRemove,
	writables copyOnWriteSet) Item {
	if item == nil {
		out := n.items[childIndex]
		childNode, removed := n.children[childIndex].pRemove(
			nil, minItems, removeMax, writables)
		n.children[childIndex], n.items[childIndex] = childNode, removed
		return out
	}
	childNode, removed := n.children[childIndex].pRemove(
		item, minItems, typ, writables)
	n.children[childIndex] = childNode
	return removed
}

// pRemove is the persistent form of remove
func (n *node) pRemove(
	item Item, minItems int, typ toRemove, writables copyOnWriteSet) (
	*node, Item) {
	wn := writables.writableNode(n)
	result := n._remove(
		item,
		minItems,
		typ,
		writables,
		pGrowChildAndRemoveShim,
		pChildRemoveShim)
	return wn, result
}

// growChildAndRemove grows child 'i' to make sure it's possible to remove an
// item from it while keeping it at minItems, then calls remove to actually
// remove it.
//
// Most documentation says we have to do two sets of special casing:
//   1) item is in this node
//   2) item is in child
// In both cases, we need to handle the two subcases:
//   A) node has enough values that it can spare one
//   B) node doesn't have enough values
// For the latter, we have to check:
//   a) left sibling has node to spare
//   b) right sibling has node to spare
//   c) we must merge
// To simplify our code here, we handle cases #1 and #2 the same:
// If a node doesn't have enough items, we make sure it does (using a,b,c).
// We then simply redo our remove call, and the second time (regardless of
// whether we're in case 1 or 2), we'll have enough items and can guarantee
// that we hit case A.
func (n *node) growChildAndRemove(i int, item Item, minItems int, typ toRemove) Item {
	child := n.children[i]
	if i > 0 && len(n.children[i-1].items) > minItems {
		// Steal from left child
		stealFrom := n.children[i-1]
		stolenItem := stealFrom.items.pop()
		child.items.insertAt(0, n.items[i-1])
		n.items[i-1] = stolenItem
		if len(stealFrom.children) > 0 {
			child.children.insertAt(0, stealFrom.children.pop())
		}
	} else if i < len(n.items) && len(n.children[i+1].items) > minItems {
		// steal from right child
		stealFrom := n.children[i+1]
		stolenItem := stealFrom.items.removeAt(0)
		child.items = append(child.items, n.items[i])
		n.items[i] = stolenItem
		if len(stealFrom.children) > 0 {
			child.children = append(child.children, stealFrom.children.removeAt(0))
		}
	} else {
		if i >= len(n.items) {
			i--
			child = n.children[i]
		}
		// merge with right child
		mergeItem := n.items.removeAt(i)
		mergeChild := n.children.removeAt(i + 1)
		child.items = append(child.items, mergeItem)
		child.items = append(child.items, mergeChild.items...)
		child.children = append(child.children, mergeChild.children...)
		n.op.freeNode(mergeChild)
	}
	return n.remove(item, minItems, typ)
}

func (n *node) pGrowChildAndRemove(i int, item Item, minItems int, typ toRemove, writables copyOnWriteSet) (*node, Item) {
	return nil, nil
}

// iterate provides a simple method for iterating over elements in the tree.
// It could probably use some work to be extra-efficient (it calls from() a
// little more than it should), but it works pretty well for now.
//
// It requires that 'from' and 'to' both return true for values we should hit
// with the iterator.  It should also be the case that 'from' returns true for
// values less than or equal to values 'to' returns true for, and 'to'
// returns true for values greater than or equal to those that 'from'
// does.
func (n *node) iterate(from, to func(Item) bool, iter ItemIterator) bool {
	for i, item := range n.items {
		if !from(item) {
			continue
		}
		if len(n.children) > 0 && !n.children[i].iterate(from, to, iter) {
			return false
		}
		if !to(item) {
			return false
		}
		if !iter(item) {
			return false
		}
	}
	if len(n.children) > 0 {
		return n.children[len(n.children)-1].iterate(from, to, iter)
	}
	return true
}

// Used for testing/debugging purposes.
func (n *node) print(w io.Writer, level int) {
	fmt.Fprintf(w, "%sNODE:%v\n", strings.Repeat("  ", level), n.items)
	for _, c := range n.children {
		c.print(w, level+1)
	}
}

type copyOnWriteSet map[*node]bool

func (s copyOnWriteSet) newNode(op *btreeOp) *node {
	result := &node{op: op}
	s[result] = true
	return result
}

func (s copyOnWriteSet) writableNode(n *node) *node {
	if s[n] {
		return n
	}
	result := s.newNode(n.op)
	result.items = append(result.items, n.items...)
	if len(n.children) > 0 {
		result.children = append(result.children, n.children...)
	}
	return result
}

type btreeOp struct {
	degree   int
	freelist *FreeList
}

// maxItems returns the max number of items to allow per node.
func (o *btreeOp) maxItems() int {
	return o.degree*2 - 1
}

// minItems returns the min number of items to allow per node (ignored for the
// root node).
func (o *btreeOp) minItems() int {
	return o.degree - 1
}

func (o *btreeOp) newNode() (n *node) {
	n = o.freelist.newNode()
	n.op = o
	return
}

func (o *btreeOp) freeNode(n *node) {
	for i := range n.items {
		n.items[i] = nil // clear to allow GC
	}
	n.items = n.items[:0]
	for i := range n.children {
		n.children[i] = nil // clear to allow GC
	}
	n.children = n.children[:0]
	n.op = nil // clear to allow GC
	o.freelist.freeNode(n)
}

// BTree is an implementation of a B-Tree.
//
// BTree stores Item instances in an ordered structure, allowing easy insertion,
// removal, and iteration.
//
// Write operations are not safe for concurrent mutation by multiple
// goroutines, but Read operations are.
type BTree struct {
	op     *btreeOp
	length int
	root   *node
}

// ReplaceOrInsert adds the given item to the tree.  If an item in the tree
// already equals the given one, it is removed from the tree and returned.
// Otherwise, nil is returned.
//
// nil cannot be added to the tree (will panic).
func (t *BTree) ReplaceOrInsert(item Item) Item {
	if item == nil {
		panic("nil item being added to BTree")
	}
	if t.root == nil {
		t.root = t.op.newNode()
		t.root.items = append(t.root.items, item)
		t.length++
		return nil
	} else if len(t.root.items) >= t.op.maxItems() {
		item2, second := t.root.split(t.op.maxItems() / 2)
		oldroot := t.root
		t.root = t.op.newNode()
		t.root.items = append(t.root.items, item2)
		t.root.children = append(t.root.children, oldroot, second)
	}
	out := t.root.insert(item, t.op.maxItems())
	if out == nil {
		t.length++
	}
	return out
}

// Delete removes an item equal to the passed in item from the tree, returning
// it.  If no such item exists, returns nil.
func (t *BTree) Delete(item Item) Item {
	return t.deleteItem(item, removeItem)
}

// DeleteMin removes the smallest item in the tree and returns it.
// If no such item exists, returns nil.
func (t *BTree) DeleteMin() Item {
	return t.deleteItem(nil, removeMin)
}

// DeleteMax removes the largest item in the tree and returns it.
// If no such item exists, returns nil.
func (t *BTree) DeleteMax() Item {
	return t.deleteItem(nil, removeMax)
}

func (t *BTree) deleteItem(item Item, typ toRemove) Item {
	if t.root == nil || len(t.root.items) == 0 {
		return nil
	}
	out := t.root.remove(item, t.op.minItems(), typ)
	if len(t.root.items) == 0 && len(t.root.children) > 0 {
		oldroot := t.root
		t.root = t.root.children[0]
		t.op.freeNode(oldroot)
	}
	if out != nil {
		t.length--
	}
	return out
}

// AscendRange calls the iterator for every value in the tree within the range
// [greaterOrEqual, lessThan), until iterator returns false.
func (t *BTree) AscendRange(greaterOrEqual, lessThan Item, iterator ItemIterator) {
	if t.root == nil {
		return
	}
	t.root.iterate(
		func(a Item) bool { return !a.Less(greaterOrEqual) },
		func(a Item) bool { return a.Less(lessThan) },
		iterator)
}

// AscendLessThan calls the iterator for every value in the tree within the range
// [first, pivot), until iterator returns false.
func (t *BTree) AscendLessThan(pivot Item, iterator ItemIterator) {
	if t.root == nil {
		return
	}
	t.root.iterate(
		func(a Item) bool { return true },
		func(a Item) bool { return a.Less(pivot) },
		iterator)
}

// AscendGreaterOrEqual calls the iterator for every value in the tree within
// the range [pivot, last], until iterator returns false.
func (t *BTree) AscendGreaterOrEqual(pivot Item, iterator ItemIterator) {
	if t.root == nil {
		return
	}
	t.root.iterate(
		func(a Item) bool { return !a.Less(pivot) },
		func(a Item) bool { return true },
		iterator)
}

// Ascend calls the iterator for every value in the tree within the range
// [first, last], until iterator returns false.
func (t *BTree) Ascend(iterator ItemIterator) {
	if t.root == nil {
		return
	}
	t.root.iterate(
		func(a Item) bool { return true },
		func(a Item) bool { return true },
		iterator)
}

// Get looks for the key item in the tree, returning it.  It returns nil if
// unable to find that item.
func (t *BTree) Get(key Item) Item {
	if t.root == nil {
		return nil
	}
	return t.root.get(key)
}

// Min returns the smallest item in the tree, or nil if the tree is empty.
func (t *BTree) Min() Item {
	return min(t.root)
}

// Max returns the largest item in the tree, or nil if the tree is empty.
func (t *BTree) Max() Item {
	return max(t.root)
}

// Has returns true if the given key is in the tree.
func (t *BTree) Has(key Item) bool {
	return t.Get(key) != nil
}

// Len returns the number of items currently in the tree.
func (t *BTree) Len() int {
	return t.length
}

// Int implements the Item interface for integers.
type Int int

// Less returns true if int(a) < int(b).
func (a Int) Less(b Item) bool {
	return a < b.(Int)
}
