# flow_prediction

A project concerned with the prediction of (laminar) velocity and pressure fields from an input shape using different U-net architectures. This work is summarized in https://arxiv.org/abs/1910.13532.

## Dataset example

The dataset contains 12.000 random shapes.

Each shape in the dataset is labeled with its velocity and pressure fields (directly as images):

<img width="944" alt="Input and predictions" src="https://user-images.githubusercontent.com/44053700/64676014-dec10f80-d474-11e9-970b-ceaf83bcef5b.png">

## Flow predictions

Here are some examples of flow predictions at ```Re=10```. Top row is the U-net prediction, while bottom row is the CFD computation.

<p align="center">
  <img width="844" alt="Prediction and baseline" src="https://user-images.githubusercontent.com/44053700/64676087-0912cd00-d475-11e9-8d50-2b3ffa012950.png">
</p>

## Architectures

Different U-net architectures were exploited in this work:

<p align="center">
  <img width="592" alt="U-net architectures" src="https://user-images.githubusercontent.com/44053700/64676009-d9fc5b80-d474-11e9-9cd4-89aa075af3c6.png">
</p>
