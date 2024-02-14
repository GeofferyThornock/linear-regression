let xs = [];
let ys = [];

let m, b;

const learningRate = 0.5;
const optimizer = tf.train.sgd(learningRate);

const loss = (pred, label) => pred.sub(label).square().mean();

function setup() {
  createCanvas(400, 400);

  m = tf.variable(tf.scalar(random(1)));
  b = tf.variable(tf.scalar(random(1)));
}


function mouseClicked() {
  let x = map(mouseX, 0, width, 0, 1);
  let y = map(mouseY, 0, height, 1, 0);
  xs.push(x);
  ys.push(y);
}

function predict(num) {
  const tfx = tf.tensor1d(num);

  const prediction = tfx.mul(m).add(b);

  return prediction;
}

function draw() {
  background(0)

  tf.tidy(() => {
    if (xs.length) {
      let tys = tf.tensor1d(ys);
      optimizer.minimize(() => loss(predict(xs), tys));
    }
  })

  stroke(255)
  strokeWeight(8)
  for (let i = 0; i < xs.length; i++) {
    let px = map(xs[i], 0, 1, 0, width);
    let py = map(ys[i], 0, 1, height, 0);

    point(px, py);
  }

  let x_predict = [0, 1];
  let y_predict = tf.tidy(() => predict(x_predict)); 
  let y_var = y_predict.dataSync();
  y_predict.dispose();

  let x1 = map(x_predict[0], 0, 1, 0, width);
  let x2 = map(x_predict[1], 0, 1, 0, width);

  let y1 = map(y_var[0], 0, 1, height, 0);
  let y2 = map(y_var[1], 0, 1, height, 0);


  line(x1, y1, x2, y2);
}
