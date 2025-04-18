export const INTRO = `Draw any digit between 0 and 9!`;

export const ABOUT = `Sea-NN is a fully on-chain convolutional neural network written in |@seahorse_lang;https://twitter.com/seahorse_lang|.

It was trained off-chain using the MNIST dataset, and then the model was uploaded on-chain. The model is quite large and takes over 200 transactions to upload (though this process could likely be optimized).

The model is largely based on the one described |here;https://victorzhou.com/blog/intro-to-cnns-part-1/|
In order to optimize the model for on-chain execution, several optimizations had to be performed 😛

- Every pixel is either 0 or 1, so a single bit can be used to represent it.
- All 512 possible 3x3 convolutions are precomputed and stored in a lookup table.
- Rather than representing weights as floats, signed integers in the range of [-2^15, 2^15] are used.
- All of the layers are optimized into a single loop, allowing the output to be computed in-place.


The entire contract is a single 73-line python file, enabled by Seahorse's brevity.
Click the "View source" button to see it 😄

Created by |@wireless_anon;https://twitter.com/wireless_anon|
Repo is available |here;https://github.com/wireless-anon/sea-nn|
`;
