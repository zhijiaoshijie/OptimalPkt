#!/bin/bash

for i in {0..11}
do
  zip -9 "/data/djl/datasets/sf11_240906_FFTFast1024_dataout/part$i.zip" -r "/data/djl/datasets/sf11_240906_FFTFast1024_dataout/part$i"
done

