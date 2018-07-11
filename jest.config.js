module.exports = {
  moduleFileExtensions: ['ts', 'js', 'json', 'node'],
  transform: {
    '\\.ts$': '<rootDir>/node_modules/ts-jest/preprocessor.js'
  },
  testRegex: '/__tests__/.*\\.(ts|js)$'
}
