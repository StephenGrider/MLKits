require('@tensorflow/tfjs-node');
const loadCSV = require('../load-csv');
const LinearRegression = require('./linear-regression');
const plot = require('node-remote-plot');

let { features, labels, testFeatures, testLabels } = loadCSV('../data/Numeric-cars-corgis.csv', {
  shuffle: true,
  splitTest: 50,
  dataColumns: [
    'Year',
    'Driveline',
    'Transmission',
    'Horsepower',
    'Torque',
    'Displacement',
    'Cylinder_Count',
    'Gears_Forward',
  ],
  labelColumns: ['MPG_CITY'],
});

const initLR = 0.1;
const regression = new LinearRegression(features, labels, {
  learningRate: initLR,
  iterations: 5,
  batchSize: 10,
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
    'Year',
    'Driveline',  (FWD, RWD, AWD, 4WD)
    'Transmission', (manual, automatic)
    'Horsepower',
    'Torque',
    'Displacement',
    'Cylinder_Count',
    'Gears_Forward',
 */
vehicles = [
  [2010, 3, 2, 350, 325, 4.2, 8, 6], // 14 mpg Audi A8
  [2009, 3, 2, 265, 243, 3.2, 6, 6], // 18 mpg Audi A5
  [2011, 2, 1, 400, 450, 4.4, 8, 6], // 17 mpg BMW 550i
  [2011, 1, 2, 108, 105, 1.6, 4, 4], // 25 mpg Chevy Aveo5 2LT AT
  [2016, 1, 2, 275, 301, 1.8, 4, 6],
];
regression.predict(vehicles).print();
