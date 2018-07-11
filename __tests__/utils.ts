import * as utils from '../src/utils'
import * as nj from 'numjs'

test('utils: norm', () => {
  const n = 10
  const vector = nj.ones(n)

  const value = utils.norm(vector)
  const expected = Math.sqrt(n)
  expect(value).toEqual(expected)
})

test('utils: tauRandInt, returns only postitve integers', () => {
  const n = 10000
  const state = nj.array([42])

  const values = Array(n).fill(0)
    .map(() => utils.tauRandInt(state))
    .filter((d) => d >= 0)

  expect(values.length).toEqual(n)
})

test('utils: tauRand, returns floats within 0 and 1', () => {
  const n = 10000
  const state = nj.array([42])

  const values = Array(n).fill(0)
    .map(() => utils.tauRand(state))
    .filter((d) => d >= 0 && d <= 1)

  expect(values.length).toEqual(n)
})

test('utils: rejectionSample, returns only unique values', () => {
  const nSamples = 100
  const poolSize = 10000
  const state = nj.array([42])

  const values = utils.rejectionSample(nSamples, poolSize, state)
  const uniqueValues = Object.keys(
    values.tolist()
      .reduce((memo, d) => ({ ...memo, [d]: true }), {})
  )

  // clearly, this is not unique
  expect(uniqueValues.length).toEqual(nSamples)
})
