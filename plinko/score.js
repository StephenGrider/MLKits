const outputs = [];
// determine the corellation between bucket and predictionPoint
const predictionPoint = 300;
const k = 3;

function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
  outputs.push([dropPosition, bounciness, size, bucketLabel]);
}

function runAnalysis() {
  const testSetSize = 10;
  const [testSet, trainingSet] = splitDataSet(outputs, testSetSize);

  let numberCorrect = 0;

  for (let i = 0; i < testSet.length; i++) {
    const bucket = knn(trainingSet, testSet[i][0]);
    if (bucket === testSet[i][3]) {
      numberCorrect++;
    }
  }

  console.log(`Accuracy: ${(numberCorrect / testSetSize) * 100}%`);
}

function knn(data, point) {
  // K-Nearest Neighbor Algorithm
  return (
    _.chain(data)
      // [[dropPosition, bucketLabel],[72,4],[227,5]]
      .map((row) => [distance(row[0], point), row[3]])
      // sort by drop position
      .sortBy((row) => row[0])
      // Gets the top 'k' results from sorted list
      .slice(0, k)
      // counts frequency of buckets
      // e.g.,  {"3":1,"4":2}
      .countBy((row) => row[1])
      // e.g., [["3",1],["4",2]]
      // converts the countBy obj to an multidimensional array
      .toPairs()
      // sorts so that the most frequent is the last array element
      .sortBy((row) => row[1])
      // get the last array element of ["bucket", frequency]
      .last()
      // e.g., "4"
      // get the bucket number (first element)
      .first()
      // e.g., 4
      // convert the string "4" to int 4
      .parseInt()
      // end the chain and return the value
      .value()
  );
}

function distance(pointA, pointB) {
  return Math.abs(pointA - pointB);
}

function splitDataSet(data, testCount) {
  const shuffled = _.shuffle(data);

  const testSet = _.slice(shuffled, 0, testCount);
  const trainingSet = _.slice(shuffled, testCount);

  console.log(`There are ${shuffled.length} total records in the sample.`);
  console.log('testSet', testSet);
  console.log('trainingSet', trainingSet);

  return [testSet, trainingSet];
}
