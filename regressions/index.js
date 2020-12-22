require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');
const LinearRegression = require('./linear-regression');

let { features, labels, testFeatures, testLabels } = loadCSV('./cars.csv', {
  shuffle: true,
  splitTest: 50,
  dataColumns: ['horsepower', 'weight', 'displacement'],
  labelColumns: ['mpg'],
});

const regression = new LinearRegression(features, labels, {
  learningRate: 0.01,
  iterations: 100,
});

regression.train();
/**
 * weights tensor has a [2,1] shape and looks like this:
 * [
 *  [0],
 *  [0]
 * ]
 */
const r2 = regression.test(testFeatures, testLabels);

console.log('R2 : ', r2);
