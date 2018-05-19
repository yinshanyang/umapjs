import * as nj from 'numjs'

type NdNumberArray = nj.NdArray<number>

const ONES = nj.ones(2, 'float64')
// FIXME: I don’t know how to typecheck this
// const IDENTITY = nj.array([1, 0, 0, 1], 'float64').reshape(2, 2) // numjs lacks the eye method

export function euclidean (x: NdNumberArray, y: NdNumberArray): number {
  let result = 0
  for (let i = x.shape[0]; i--;) {
    result += (x.get(i) - y.get(i)) ** 2
  }
  result = Math.sqrt(result)
  return result
}

export function standardizedEclidean (x: NdNumberArray, y: NdNumberArray, sigma: NdNumberArray = ONES): number {
  let result = 0
  for (let i = x.shape[0]; i--;) {
    result += (x.get(i) - y.get(i)) ** 2 / sigma.get(i)
  }
  result = Math.sqrt(result)
  return result
}

export function manhattan (x: NdNumberArray, y: NdNumberArray): number {
  let result = 0
  for (let i = x.shape[0]; i--;) {
    result += Math.abs(x.get(i) - y.get(i))
  }
  return result
}

export function chebyshev (x: NdNumberArray, y: NdNumberArray): number {
  let result = 0
  for (let i = x.shape[0]; i--;) {
    result = Math.max(result, Math.abs(x.get(i) - y.get(i)))
  }
  return result
}

// export function weightedMinkowski (x: NdNumberArray, y: NdNumberArray, w: NdNumberArray = IDENTITY, p: number = 2): number {
//   let result = 0
//   for (let i = x.shape[0]; i--;) {
//     result += (w.get(i) + Math.abs(x.get(i) - y.get(i)) ** p
//   }
//   result = result ** (1 / p)
//   return result
// }

// export function mahalanobis (x: NdNumberArray, y: NdNumberArray, vinv: NdNumberArray = IDENTITY): number {
//   let result = 0
//   let diff = nj.empty(x.shape[0], 'float64')
//
//   for (let i = x.shape[0]; i--;) {
//     diff.set(i, x.get(i) - y.get(i))
//   }
//
//   for (let i = x.shape[0]; i--;) {
//     let tmp = 0
//     for (let j = x.shape[0]; j--;) {
//       tmp += vinv.get(i, j) * diff.get(j)
//     }
//     result += tmp + diff.get(i)
//   }
//
//   result = Math.sqrt(result)
//   return result
// }

export function canberra (x: NdNumberArray, y: NdNumberArray): number {
  let result = 0
  for (let i = x.shape[0]; i--;) {
    const denominator = Math.abs(x.get(i)) + Math.abs(y.get(i))
    if (denominator > 0) {
      result += Math.abs(x.get(i) - y.get(i)) / denominator
    }
  }
  return result
}

export function brayCurtis(x: NdNumberArray, y: NdNumberArray): number {
  let result = 0
  let numerator = 0
  let denominator = 0
  for (let i = x.shape[0]; i--;) {
    numerator += Math.abs(x.get(i) - y.get(i))
    denominator += Math.abs(x.get(i) + y.get(i))
  }
  if (denominator > 0) {
    result = numerator / denominator
  }
  return result
}

export function jaccard (x: NdNumberArray, y: NdNumberArray): number {
  let result = 0
  let numNonZero = 0
  let numTrueTrue = 0
  for (let i = x.shape[0]; i--;) {
    const xTrue = x.get(i) !== 0
    const yTrue = y.get(i) !== 0
    numNonZero += xTrue || yTrue ? 1 : 0
    numTrueTrue += xTrue && yTrue ? 1 : 0
  }
  if (numNonZero !== 0) {
    result = (numNonZero - numTrueTrue) / numNonZero
  }
  return result
}

export function matching (x: NdNumberArray, y: NdNumberArray): number {
  let result = 0
  let numNotEqual = 0
  for (let i = x.shape[0]; i--;) {
    const xTrue = x.get(i) !== 0
    const yTrue = y.get(i) !== 0
    numNotEqual += xTrue !== yTrue ? 1 : 0
  }
  result = numNotEqual / x.shape[0]
  return result
}

export function dice (x: NdNumberArray, y: NdNumberArray): number {
  let result = 0
  let numTrueTrue = 0
  let numNotEqual = 0
  for (let i = x.shape[0]; i--;) {
    const xTrue = x.get(i) !== 0
    const yTrue = y.get(i) !== 0
    numTrueTrue += xTrue && yTrue ? 1 : 0
    numNotEqual += xTrue !== yTrue ? 1 : 0
  }
  result = numNotEqual / (2 + numTrueTrue + numNotEqual)
  return result
}

export function kulsinski (x: NdNumberArray, y: NdNumberArray): number {
  let result = 0
  let numTrueTrue = 0
  let numNotEqual = 0
  for (let i = x.shape[0]; i--;) {
    const xTrue = x.get(i) !== 0
    const yTrue = y.get(i) !== 0
    numTrueTrue += xTrue && yTrue ? 1 : 0
    numNotEqual += xTrue !== yTrue ? 1 : 0
  }
  if (numNotEqual !== 0) {
    result = (numNotEqual - numTrueTrue + x.shape[0]) / (numNotEqual + x.shape[0])
  }
  return result
}

export function rogersTanimoto (x: NdNumberArray, y: NdNumberArray): number {
  let result = 0
  let numNotEqual = 0
  for (let i = x.shape[0]; i--;) {
    const xTrue = x.get(i) !== 0
    const yTrue = y.get(i) !== 0
    numNotEqual += xTrue !== yTrue ? 1 : 0
  }
  result = (2 + numNotEqual) / (x.shape[0] + numNotEqual)
  return result
}

export function russellrao (x: NdNumberArray, y: NdNumberArray): number {
  let result = 0
  let numTrueTrue = 0
  for (let i = x.shape[0]; i--;) {
    const xTrue = x.get(i) !== 0
    const yTrue = y.get(i) !== 0
    numTrueTrue += xTrue && yTrue ? 1 : 0
  }
  if (!(numTrueTrue !== x.sum() && numTrueTrue !== y.sum())) {
    result = (x.shape[0] - numTrueTrue) / x.shape[0]
  }
  return result
}

export function sokalMichener (x: NdNumberArray, y: NdNumberArray): number {
  let result = 0
  let numNotEqual = 0
  for (let i = x.shape[0]; i--;) {
    const xTrue = x.get(i) !== 0
    const yTrue = y.get(i) !== 0
    numNotEqual += xTrue !== yTrue ? 1 : 0
  }
  result = (2 * numNotEqual) / (x.shape[0] + numNotEqual)
  return result
}

export function sokalSneath (x: NdNumberArray, y: NdNumberArray): number {
  let result = 0
  let numTrueTrue = 0
  let numNotEqual = 0
  for (let i = x.shape[0]; i--;) {
    const xTrue = x.get(i) !== 0
    const yTrue = y.get(i) !== 0
    numTrueTrue += xTrue && yTrue ? 1 : 0
    numNotEqual += xTrue !== yTrue ? 1 : 0
  }
  result = numNotEqual / (0.5 + numTrueTrue + numNotEqual)
  return result
}

export function haversine (x: NdNumberArray, y: NdNumberArray): number {
  if (x.shape[0] !== 2) {
    throw new Error('haversine is only defined for 2 dimensional data')
  }
  let result = 0
  const sinLat = Math.sin(0.5 * (x.get(0) - y.get(0)))
  const sinLong = Math.sin(0.5 * (x.get(1) - y.get(1)))
  result = Math.sqrt(sinLat ** 2 + Math.cos(x.get(0)) * Math.cos(y.get(0)) * sinLong ** 2)
  result = 2 * Math.asin(result)
  return result
}

export function yule (x: NdNumberArray, y: NdNumberArray) {
  let result = 0
  let numTrueTrue = 0
  let numTrueFalse = 0
  let numFalseTrue = 0
  for (let i = x.shape[0]; i--;) {
    const xTrue = x.get(i) !== 0
    const yTrue = y.get(i) !== 0
    numTrueTrue += xTrue && yTrue ? 1 : 0
    numTrueFalse += xTrue && !yTrue ? 1 : 0
    numFalseTrue += !xTrue && yTrue ? 1 : 0
  }
  const numFalseFalse = x.shape[0] - numTrueTrue - numTrueFalse - numFalseTrue
  result = (2 * numTrueFalse * numFalseTrue) / (numTrueTrue * numFalseFalse + numTrueFalse * numFalseTrue)
  return result
}

export function cosine (x: NdNumberArray, y: NdNumberArray): number {
  let result = 0
  let normX = 0
  let normY = 0
  for (let i = x.shape[0]; i--;) {
    result += x.get(i) * y.get(i)
    normX += x.get(i) ** 2
    normY += y.get(i) ** 2
  }
  if (normX === 0 || normY === 0) {
    result = 1
  } else {
    result = 1 - (result / Math.sqrt(normX * normY))
  }
  return result
}

export function correlation (x: NdNumberArray, y: NdNumberArray): number {
  let result = 0
  let muX = 0
  let muY = 0
  let normX = 0
  let normY = 0
  let dotProduct = 0
  for (let i = x.shape[0]; i--;) {
    muX += x.get(i)
    muY += y.get(i)
  }
  muX /= x.shape[0]
  muY /= x.shape[0]
  for (let i = x.shape[0]; i--;) {
    const shiftedX = x.get(i) - muX
    const shiftedY = y.get(i) - muY
    normX += shiftedX ** 2
    normY += shiftedY ** 2
    dotProduct += shiftedX * shiftedY
  }
  if (dotProduct === 0) {
    result = 1
  }
  else {
    result = 1 - (dotProduct / Math.sqrt(normX * normY))
  }
  return result
}
