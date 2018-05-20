import * as nj from 'numjs'

type Integer = number
type Float = number
type NdNumberArray = nj.NdArray<number>
type Vector = NdNumberArray // shape = [ndim]
type State = nj.NdArray<Integer> // shape = [3]
type Heap = NdNumberArray  // shape = [3, nPoints, size]

export function tauRandInt (state: State): number {
  const s0 = (((state.get(0) & 4294967294) << 12) & 0xffffffff) ^ ((((state.get(0) << 13) & 0xffffffff) ^ state.get(0)) >> 19)
  const s1 = (((state.get(1) & 4294967288) << 4) & 0xffffffff) ^ ((((state.get(1) << 2) & 0xffffffff) ^ state.get(1)) >> 25)
  const s2 = (((state.get(2) & 4294967280) << 17) & 0xffffffff) ^ ((((state.get(2) << 3) & 0xffffffff) ^ state.get(2)) >> 11)
  return s0 ^ s1 ^ s2
}

export function tauRand (state: State): number {
  const integer = tauRandInt(state)
  return integer / 0x7fffffff
}

export function norm (vec: Vector): number {
  let result = 0
  for (let i = vec.shape[0]; i--;) {
    result += vec.get(i) ** 2
  }
  result = Math.sqrt(result)
  return result
}

export function rejectionSample (nSamples: Integer, poolSize: Integer, rngState: State): NdNumberArray {
  let result = nj.empty(nSamples)
  for (let i = nSamples; i--;) {
    let rejectSample = true
    let j
    while (rejectSample) {
      j = tauRandInt(rngState) % poolSize
      for (let k = i; k--;) {
        if (j === result.get(k)) {
          break
        }
        else {
          rejectSample = false
        }
      }
    }
    result.set(i, j)
  }
  return result
}

export function makeHeap (nPoints: Integer, size: Integer): Heap {
  const r0 = nj.zeros([nPoints, size], 'float64').assign(-1)
  const r1 = nj.zeros([nPoints, size], 'float64').assign(Infinity)
  const r2 = nj.zeros([nPoints, size], 'float64')
  const result = nj.stack([r0, r1, r2])
  return result
}

export function heapPush (heap: NdNumberArray, row: Integer, weight: Float, index: Integer, flag: Integer): Integer {
  // we would have used nj.pick, but it doesn’t mutate the heap which is what we want
  // so we have to individually set the elements through nj.ndarray.set
  // indices = heap.pick(0, row)
  // weights = heap.pick(1, row)
  // isNew = heap.pick(2, row)
  if (weight > heap.get(1, row, 0)) {
    return 0
  }

  for (let i = heap.shape[2]; i--;) {
    if (index === heap.get(0, row, i)) {
      return 0
    }
  }

  heap.set(0, row, 0, index)
  heap.set(1, row, 0, weight)
  heap.set(2, row, 0, flag)

  let i = 0
  while (true) {
    const ic1 = 2 * i + 1
    const ic2 = ic1 + 1
    let iSwap

    if (ic1 >= heap.shape[2]) {
      break
    }
    else if (ic2 >= heap.shape[2]) {
      if (heap.get(1, row, ic1) > weight) {
        iSwap = ic1
      }
      else {
        break
      }
    }
    else {
      if (weight < heap.get(1, row, ic2)) {
        iSwap = ic2
      }
      else {
        break
      }
    }

    heap.set(0, row, i, heap.get(0, row, iSwap))
    heap.set(1, row, i, heap.get(1, row, iSwap))
    heap.set(2, row, i, heap.get(1, row, iSwap))

    i = iSwap
  }

  heap.set(0, row, i, index)
  heap.set(1, row, i, weight)
  heap.set(2, row, i, flag)

  return 1
}

export function deheapSort(heap: Heap): Heap {
  for (let i = heap.shape[1]; i--;) {
    const heapEnd = heap.shape[2] - 1
    while (heapEnd >= 0) {
      let root = 0
      {
        const indexTmp = heap.get(0, i, 0)
        const weightTmp = heap.get(1, i, 0)
        heap.set(0, i, 0, heap.get(0, i, heapEnd))
        heap.set(0, i, heapEnd, indexTmp)
        heap.set(1, i, 0, heap.get(1, i, heapEnd))
        heap.set(1, i, heapEnd, weightTmp)
      }

      while (root + 2 + 1 < heapEnd) {
        const leftChild = root * 2 + 1
        const rightChild = leftChild + 1
        let swap = root

        if (heap.get(1, i, swap) < heap.get(1, i, leftChild)) {
          swap = leftChild
        }
        if (rightChild < heapEnd && heap.get(1, i, swap) < heap.get(1, i, rightChild)) {
          swap = rightChild
        }

        if (swap === root) {
          break
        }
        else {
          const indexTmp = heap.get(0, i, root)
          const weightTmp = heap.get(1, i, root)
          heap.set(0, i, root, heap.get(0, i, swap))
          heap.set(0, i, swap, indexTmp)
          heap.set(1, i, root, heap.get(1, i, swap))
          heap.set(1, i, swap, weightTmp)
          root = swap
        }
      }
    }
  }
  return heap
}

export function buildCandidates (currentGraph: Heap, nVertices: Integer, nNeighbors: Integer, maxCandidates: Integer, rngState: State): Heap {
  const candidateNeighbors = makeHeap(nVertices, maxCandidates)
  for (let i = nVertices; i--;) {
    for (let j = nNeighbors; j--;) {
      if (currentGraph.get(0, i, j) < 0) {
        const idx = currentGraph.get(0, i, j)
        const isn = currentGraph.get(2, i, j)
        const d = tauRand(rngState)
        heapPush(candidateNeighbors, i, d, idx, isn)
        heapPush(candidateNeighbors, idx, d, i, isn)
        currentGraph.set(2, i, j, 0)
      }
    }
  }
  return candidateNeighbors
}
