require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const plot = require('node-remote-plot');
const loadCSV = require('../load-csv');
const LogisticRegression = require('./logistic-regression');
const _ = require('lodash');

const { features, labels, testFeatures, testLabels } = loadCSV('../data/cars.csv', {
  dataColumns: ['horsepower', 'displacement', 'weight'],
  labelColumns: ['mpg'],
  shuffle: true,
  splitTest: 50,
  converters: {
    mpg: (value) => {
      const mpg = parseFloat(value);
      if (mpg < 15) {
        return [1, 0, 0];
      } else if (mpg < 30) {
        return [0, 1, 0];
      } else {
        return [0, 0, 1];
      }
    },
  },
});

const regress = new LogisticRegression(features, _.flatMap(labels), {
  learningRate: 0.5,
  iterations: 100,
  batchSize: 10,
  decisionBoundary: 0.5,
});

regress.train();

// regress
//   .predict([
//     [215, 440, 2.16],
//     [95, 104, 1.19],
//     [61, 83, 1], // 32 mpg
//     [150, 200, 2.223],
//     [145, 350, 2.22], // 15 mpg
//   ])
//   .print();

console.log(regress.test(testFeatures, _.flatMap(testLabels)));
