require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');
const LinearRegression = require('./linear-regression');
const plot = require('node-remote-plot');
const { initial } = require('lodash');

let { features, labels, testFeatures, testLabels } = loadCSV('./cars.csv', {
  shuffle: true,
  splitTest: 50,
  dataColumns: ['horsepower', 'weight', 'displacement'],
  labelColumns: ['mpg'],
});

const initLR = 0.1;
const regression = new LinearRegression(features, labels, {
  learningRate: initLR,
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
plot({
  x: regression.mseHistory.reverse(),
  xLabel: 'Iterations',
  yLabel: 'MSE',
});
console.log('R2 : ', r2, ' initLR: ', initLR, ' iterations: ', regression.options.iterations);
