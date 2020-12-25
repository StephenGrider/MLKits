require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');
const LinearRegression = require('./linear-regression');
const plot = require('node-remote-plot');
const { initial } = require('lodash');

let { features, labels, testFeatures, testLabels } = loadCSV('./cars.csv', {
  shuffle: true,
  splitTest: 50,
  dataColumns: ['displacement', 'horsepower', 'weight', 'acceleration'],
  labelColumns: ['mpg'],
});

const initLR = 0.1;
const regression = new LinearRegression(features, labels, {
  learningRate: initLR,
  iterations: 30,
  batchSize: 1,
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

litersToCID = (liters) => {
  // There are 61 cubic inches in a liter
  return liters * 61;
};

/**
 * mpg, cyl, displacement, hp, wt, acc
 * [13,8,400,175,2.57,12],
 * [11,8,400,150,2.5,14],
 * [12,8,383,180,2.48,11],
 * [12,8,429,198,2.48,11],
 * [12,8,455,225,2.48,11],
 * [12,8,400,167,2.45,12],
 * [13,8,400,170,2.37,12],
 */
vehicles = [
  [400, 175, 2.57, 12],
  [400, 150, 2.5, 14],
  [383, 180, 2.48, 11],
  [429, 198, 2.48, 11],
  [455, 225, 2.48, 11],
  [400, 167, 2.45, 12],
  [400, 170, 2.37, 12],
];
regression.predict(vehicles).print();
