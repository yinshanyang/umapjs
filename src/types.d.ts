import * as nj from 'numjs'

export type Integer = number
export type Float = number
export type NdNumberArray = nj.NdArray<number>
export type Vector = NdNumberArray // shape = [ndim]
export type State = nj.NdArray<Integer> // shape = [3]
export type Heap = NdNumberArray  // shape = [3, nPoints, size]
